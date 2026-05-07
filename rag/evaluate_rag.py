import asyncio
import os
from typing import Union, List, Optional, Type

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from config import VECTORSTORE_DIR, MODEL_NAME, TOP_K
from pipeline import load_vectorstore

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# output
EVAL_OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qa_eval_results.csv"
)


# LLM utility 
async def run_llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    model: str = MODEL_NAME,
) -> Union[str, BaseModel, None]:
    
    max_retries = 5
    base_delay  = 10

    for attempt in range(1, max_retries + 1):
        try:
            if response_model:
                response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user",   "content": user_prompt.strip()},
                    ],
                    temperature=0,
                    response_format=response_model,
                )
                return response.choices[0].message.parsed
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user",   "content": user_prompt.strip()},
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"  [LLM attempt {attempt}/{max_retries}] Error: {e}")
            if attempt == max_retries:
                return None
            await asyncio.sleep(300)


async def run_llm_calls(
    system_prompts: List[str],
    user_prompts: List[str],
    response_model: Optional[Type[BaseModel]] = None,
    model: str = MODEL_NAME,
) -> List:
    """parallel LLM calls"""
    tasks = [
        run_llm_call(sp, up, response_model=response_model, model=model)
        for sp, up in zip(system_prompts, user_prompts)
    ]
    return await asyncio.gather(*tasks)


# 1- extract chunks + generate Q&A

class QAPair(BaseModel):
    question:   str
    answer:     str   # ground truth
    chunk_used: str   # chunk source


class QAList(BaseModel):
    pairs: List[QAPair]


SYSTEM_QA_GENERATION = """
You are an expert at creating evaluation datasets from scientific article excerpts.

Given a passage from a scientific article, generate {n_questions} question/answer pairs

Rules:
- Questions must be precise and answerable from the passage alone
- Answers must be factually grounded in the passage (no inference beyond the text)
- Vary question types: factual, comparative, causal, definitional
- Return valid JSON only
"""

USER_QA_GENERATION = """
Article passage:

{chunk}

Generate {n_questions} question/answer pairs from this passage.
"""


def get_all_chunks_from_faiss() -> List[str]:
    """
    load all text chunks stored in the FAISS vector store
    """
    vectorstore = load_vectorstore()
    docs   = list(vectorstore.docstore._dict.values())
    chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
    print(f"{len(chunks)} chunks extracted from faiss")
    return chunks


async def generate_qa_from_vectorstore(
    n_questions_per_chunk: int = 3,
    max_chunks: int = 50,
) -> pd.DataFrame:
    """
    generate Q&A dataset from FAISS chunks

    Args:
        n_questions_per_chunk: number of Q&A generated per chunk
        max_chunks: limit to prevent too many LLM calls

    Returns:
        df with values for question, answer_ref, context
    """
    chunks = get_all_chunks_from_faiss()[:max_chunks]
    sys_prompts  = [SYSTEM_QA_GENERATION.format(n_questions=n_questions_per_chunk)] * len(chunks)
    user_prompts = [
        USER_QA_GENERATION.format(chunk=chunk, n_questions=n_questions_per_chunk)
        for chunk in chunks
    ]

    responses = await run_llm_calls(sys_prompts, user_prompts, response_model=QAList)

    rows = []
    for resp in responses:
        if resp:
            for pair in resp.pairs:
                rows.append({
                    "question":   pair.question,
                    "answer_ref": pair.answer,
                    "context":    pair.chunk_used,
                })

    df = pd.DataFrame(rows)
    print(f"{len(df)} Q&A generated pairs")
    return df


# 2- RAG answers

async def generate_agent_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    call oceanAI rag to answer the generated questions 
    Returns:
        previous dataframe + context_retrieved, answer_rag
    """
    from main import build_rag_chain

    vectorstore = load_vectorstore()
    chain       = build_rag_chain(vectorstore)
    answers  = []
    contexts = []

    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"  question {i}/{len(df)}")
        response = chain.invoke({"query": row["question"]})
        answers.append(response["result"])
        context = "\n\n---\n\n".join(
            doc.page_content for doc in response["source_documents"]
        )
        contexts.append(context)

    df = df.copy()
    df["answer_rag"]      = answers
    df["context_retrieved"] = contexts
    print('generating Q&A done')
    return df


# 3- Evaluation with LLM as a judge

class CompletenessScore(BaseModel):
    reasoning:   str
    is_complete: bool
    score:       float   #0 to 1


SYSTEM_COMPLETENESS = """
You are evaluating whether an agent's answer to a scientific question is complete.

A complete answer:
- Covers all key points present in the reference answer
- Does not omit important facts or concepts
- Is based on the provided context

Respond with:
- reasoning: brief explanation
- is_complete: true or false
- score: float between 0 and 1
"""

USER_COMPLETENESS = """
Question: {question}

Reference answer (ground truth): {answer_ref}

Agent answer: {answer_rag}

Context: {context}
"""


async def evaluate_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates the completeness of each rag response vs the ground truth"""

    context_col = "context_retrieved" if "context_retrieved" in df.columns else "context"

    sys_msgs  = [SYSTEM_COMPLETENESS] * len(df)
    user_msgs = [
        USER_COMPLETENESS.format(
            question=row["question"],
            answer_ref=row["answer_ref"],
            answer_rag=row["answer_rag"],
            context=row[context_col],
        )
        for _, row in df.iterrows()
    ]

    responses = await run_llm_calls(sys_msgs, user_msgs, response_model=CompletenessScore)

    df = df.copy()
    df["completeness_score"]     = [r.score     if r else None for r in responses]
    df["completeness_reasoning"] = [r.reasoning if r else None for r in responses]
    print('completeness evaluation done')
    return df


# 4- evaluaiton with RAGAS

def evaluate_ragas_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate ragas metrics : faithfulness, relevance and faithfulness
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.dataset_schema import SingleTurnSample
    from langchain_openai import ChatOpenAI

    langchain_llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    context_col = "context_retrieved" if "context_retrieved" in df.columns else "context"

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer_rag"],
            retrieved_contexts=[row[context_col]],
            reference=row["answer_ref"],
        )
        for _, row in df.iterrows()
        if row["answer_rag"]
    ]

    dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=langchain_llm,
    )

    df = df.copy()
    df["faithfulness"]      = [r.get("faithfulness")      for r in results.scores]
    df["answer_relevancy"]  = [r.get("answer_relevancy")  for r in results.scores]
    df["context_precision"] = [r.get("context_precision") for r in results.scores]

    print('ragas evaluation done')
    return df


async def run_full_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    df = await evaluate_completeness(df)
    df = evaluate_ragas_metrics(df)
    return df


async def main(
    n_questions_per_chunk: int = 3,
    max_chunks: int = 50,
    output_csv: str = EVAL_OUTPUT_CSV,
):
    
    # generate Q&A
    df = await generate_qa_from_vectorstore(
        n_questions_per_chunk=n_questions_per_chunk,
        max_chunks=max_chunks,
    )

    # get rag answers
    df = await generate_agent_answers(df)

    # evaluation
    df = await run_full_evaluation(df)

    df.to_csv(output_csv, index=False)
    print(f"\n[Done] results saved in : {output_csv}")
    print(df[[
        "question", "completeness_score",
        "faithfulness", "answer_relevancy", "context_precision"
    ]].to_string(index=False))

    return df


if __name__ == "__main__":
    asyncio.run(main(
        n_questions_per_chunk=3,
        max_chunks=5,
    ))