# ============================================================
# evaluation/evaluation.py
# ============================================================
# Pipeline complet : Génération Q&A + Évaluation multi-métriques
#
# Structure :
#   Module 1 — Génération Q&A depuis les chunks du vectorstore FAISS
#   Module 2 — Réponses RAG (retrieval + génération)
#   Module 3 — Évaluation : complétude (LLM-judge) + RAGAS
#   Module 4 — Benchmark vs annotations humaines (optionnel)
#
# Usage :
#   cd rag/
#   python evaluation/evaluation.py
# ============================================================

import asyncio
import os
import sys
import random
from typing import Union, List, Optional, Type

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# UTILITAIRE LLM — appels async avec retry exponentiel

async def run_llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    model: str = MODEL_NAME,
) -> Union[str, BaseModel, None]:
    """
    Appel LLM unique avec retry exponentiel.
    - response_model fourni → sortie structurée Pydantic
    - sinon → texte brut
    """
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
                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  [LLM attempt {attempt}/{max_retries}] Error: {e}")
            if attempt == max_retries:
                return None
            wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait)


async def run_llm_calls(
    system_prompts: List[str],
    user_prompts: List[str],
    response_model: Optional[Type[BaseModel]] = None,
    model: str = MODEL_NAME,
) -> List:
    """Appels LLM en parallèle."""
    tasks = [
        run_llm_call(sp, up, response_model=response_model, model=model)
        for sp, up in zip(system_prompts, user_prompts)
    ]
    return await asyncio.gather(*tasks)


# MODULE 1 — GÉNÉRATION Q&A depuis le vectorstore FAISS

class QAPair(BaseModel):
    question:   str
    answer:     str   # ground truth
    chunk_used: str   # chunk source


class QAList(BaseModel):
    pairs: List[QAPair]


SYSTEM_QA_GENERATION = """
You are an expert at creating evaluation datasets from scientific article excerpts.

Given a passage from a scientific article, generate {n_questions} question-answer pairs.

Rules:
- Questions must be precise and answerable from the passage alone
- Answers must be factually grounded in the passage (no inference beyond the text)
- Vary question types: factual, comparative, causal, definitional
- Return valid JSON only
""".strip()

USER_QA_GENERATION = """
Article passage:

{chunk}

Generate {n_questions} question-answer pairs from this passage.
""".strip()


def get_all_chunks_from_faiss() -> List[str]:
    """
    Extrait tous les chunks texte stockés dans le vectorstore FAISS.
    Réutilise load_vectorstore() de pipeline.py — pas de re-parsing PDF.
    """
    vectorstore = load_vectorstore()
    docs   = list(vectorstore.docstore._dict.values())
    chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
    print(f"[Module 1] {len(chunks)} chunks extraits du vectorstore FAISS.")
    return chunks


async def generate_qa_from_vectorstore(
    n_questions_per_chunk: int = 3,
    max_chunks: int = 50,
) -> pd.DataFrame:
    """
    Module 1 : génère un dataset Q&A depuis les chunks FAISS existants.
    Aucun reparsing PDF, les chunks sont déjà propres

    Args:
        n_questions_per_chunk : nb de Q&A générées par chunk
        max_chunks            : limite pour éviter trop d'appels LLM

    Returns:
        DataFrame avec colonnes : question, answer_ref, context
    """
    chunks = get_all_chunks_from_faiss()[:max_chunks]
    print(f"[Module 1] Génération Q&A sur {len(chunks)} chunks...")

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
    print(f"[Module 1] {len(df)} paires Q&A générées.")
    return df


# MODULE 2 — RÉPONSES RAG (retrieval FAISS + génération)

SYSTEM_AGENT = """
You are a scientific assistant answering questions based strictly on the provided context.

Rules:
- Base your answer only on the context below
- If the context doesn't contain the answer, say "Not found in context"
- Be precise and concise
"""

USER_AGENT = """
Context:
{context}

Question:
{question}
"""


def retrieve_context(vectorstore, question: str, k: int = TOP_K) -> str:
    """Récupère les k chunks les plus proches et les concatène."""
    docs = vectorstore.similarity_search(question, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


async def generate_agent_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 2 : pour chaque question, récupère le contexte FAISS
    et génère la réponse agent.

    Returns:
        DataFrame enrichi avec : context_retrieved, answer_agent
    """
    vectorstore = load_vectorstore()
    print(f"[Module 2] Retrieval + génération pour {len(df)} questions (k={TOP_K})...")

    retrieved_contexts = [
        retrieve_context(vectorstore, row["question"])
        for _, row in df.iterrows()
    ]

    sys_prompts  = [SYSTEM_AGENT] * len(df)
    user_prompts = [
        USER_AGENT.format(context=ctx, question=row["question"])
        for ctx, (_, row) in zip(retrieved_contexts, df.iterrows())
    ]

    responses = await run_llm_calls(sys_prompts, user_prompts)

    df = df.copy()
    df["context_retrieved"] = retrieved_contexts
    df["answer_agent"]      = responses
    return df


# MODULE 3a — COMPLÉTUDE (LLM-as-judge)

class CompletenessScore(BaseModel):
    reasoning:   str
    is_complete: bool
    score:       float   # 0.0 à 1.0


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
""".strip()

USER_COMPLETENESS = """
Question: {question}

Reference answer (ground truth): {answer_ref}

Agent answer: {answer_agent}

Context: {context}
""".strip()


async def evaluate_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Évalue la complétude de chaque réponse agent vs ground truth."""
    print("[Module 3a] Évaluation complétude...")

    context_col = "context_retrieved" if "context_retrieved" in df.columns else "context"

    sys_msgs  = [SYSTEM_COMPLETENESS] * len(df)
    user_msgs = [
        USER_COMPLETENESS.format(
            question=row["question"],
            answer_ref=row["answer_ref"],
            answer_agent=row["answer_agent"],
            context=row[context_col],
        )
        for _, row in df.iterrows()
    ]

    responses = await run_llm_calls(sys_msgs, user_msgs, response_model=CompletenessScore)

    df = df.copy()
    df["completeness_score"]     = [r.score     if r else None for r in responses]
    df["completeness_reasoning"] = [r.reasoning if r else None for r in responses]
    return df


# MODULE 3b — RAGAS (faithfulness, relevancy, context_precision)

def evaluate_ragas_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Évalue 3 métriques RAGAS :
    - faithfulness      : réponse fondée sur le contexte ?
    - answer_relevancy  : réponse pertinente par rapport à la question ?
    - context_precision : contexte récupéré pertinent ?
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.dataset_schema import SingleTurnSample
    from langchain_openai import ChatOpenAI

    print("[Module 3b] Évaluation RAGAS (faithfulness, relevancy, precision)...")

    langchain_llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    context_col = "context_retrieved" if "context_retrieved" in df.columns else "context"

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer_agent"],
            retrieved_contexts=[row[context_col]],
            reference=row["answer_ref"],
        )
        for _, row in df.iterrows()
        if row["answer_agent"]
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
    return df


async def run_full_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Lance complétude (async) + RAGAS (sync)."""
    df = await evaluate_completeness(df)
    df = evaluate_ragas_metrics(df)
    return df


# MODULE 4 — BENCHMARK vs annotations humaines (optionnel)

def benchmark_metrics(df: pd.DataFrame, human_annotation_col: str = "human_label") -> dict:
    """
    Compare chaque métrique automatique vs annotation humaine.

    Args:
        df                   : DataFrame avec métriques + colonne annotation
        human_annotation_col : colonne contenant 1 (good) / 0 (bad)

    Returns:
        dict avec classification_report par métrique
    """
    from sklearn.metrics import classification_report

    if human_annotation_col not in df.columns:
        print("[Module 4] Pas d'annotation humaine trouvée — benchmark ignoré.")
        return {}

    print(f"[Module 4] Benchmark des métriques vs '{human_annotation_col}'...")

    thresholds = {
        "completeness_score": 0.5,
        "faithfulness":       0.5,
        "answer_relevancy":   0.5,
        "context_precision":  0.5,
    }
    reports = {}

    for metric, threshold in thresholds.items():
        if metric not in df.columns:
            continue
        sub = df[[human_annotation_col, metric]].dropna()
        if len(sub) == 0:
            continue
        predicted = (sub[metric] >= threshold).astype(int)
        report    = classification_report(
            sub[human_annotation_col], predicted,
            output_dict=True, zero_division=0,
        )
        reports[metric] = report
        f1 = report.get("weighted avg", {}).get("f1-score", "N/A")
        print(f"  {metric:<25} → F1 = {f1:.3f}" if isinstance(f1, float) else f"  {metric}: N/A")

    return reports


# ============================================================
# MAIN
# ============================================================

async def main(
    n_questions_per_chunk: int = 3,
    max_chunks: int = 50,
    output_csv: str = EVAL_OUTPUT_CSV,
):
    """
    Pipeline complet depuis le vectorstore FAISS existant :
      1. Extrait les chunks + génère Q&A (ground truth)
      2. Retrieval FAISS + génère réponses agent
      3. Évalue : complétude + faithfulness + relevancy + context_precision
      4. Benchmark si annotations humaines disponibles
      5. Sauvegarde CSV dans evaluation/
    """
    # 1. Génération Q&A
    df = await generate_qa_from_vectorstore(
        n_questions_per_chunk=n_questions_per_chunk,
        max_chunks=max_chunks,
    )

    # 2. Réponses agent
    df = await generate_agent_answers(df)

    # 3. Évaluation
    df = await run_full_evaluation(df)

    # 4. Benchmark (si annotations humaines)
    if "human_label" in df.columns:
        benchmark_metrics(df)

    # 5. Sauvegarde
    df.to_csv(output_csv, index=False)
    print(f"\n[Done] Résultats sauvegardés dans : {output_csv}")
    print("\n--- Aperçu des scores ---")
    print(df[[
        "question", "completeness_score",
        "faithfulness", "answer_relevancy", "context_precision"
    ]].to_string(index=False))

    return df


if __name__ == "__main__":
    asyncio.run(main(
        n_questions_per_chunk=3,
        max_chunks=50,
    ))