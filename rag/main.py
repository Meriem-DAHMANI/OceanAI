
import os
import sys
import argparse
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config import PAPERS_DIR, MODEL_NAME, TOP_K
from pipeline import (
    load_and_clean_pdf, get_document_title, split_into_chunks,
    token_length, build_and_save_vectorstore, load_vectorstore,
    vectorstore_exists,
)


def process_all_papers(papers_dir: str) -> list[dict]:
    pdf_paths = sorted([
        os.path.join(papers_dir, f)
        for f in os.listdir(papers_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_paths:
        print(f"No PDFs found in: {papers_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s) in {papers_dir}\n")
    all_chunks = []

    for i, path in enumerate(pdf_paths, 1):
        print(f"[{i}/{len(pdf_paths)}] {os.path.basename(path)}")

        text = load_and_clean_pdf(path)
        if not text:
            print(" Skipping — no extractable text\n")
            continue

        title  = get_document_title(text)
        print(f" Title: {title}")

        chunks = split_into_chunks(text, document_title=title)
        avg    = sum(token_length(c["text"]) for c in chunks) / len(chunks)
        print(f"{len(chunks)} chunks, avg {avg:.0f} tokens\n")

        all_chunks.extend(chunks)

    print(f"Total chunks across all papers: {len(all_chunks)}")
    return all_chunks


class RagAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")
    from_papers: bool = Field(
        description="True if the answer is based on the paper excerpts, "
                    "False if it comes from general knowledge"
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are OceanAI, a marine science assistant.\n"
     "You are given excerpts from scientific papers.\n\n"
     "- If the excerpts contain the answer, answer using ONLY the excerpts "
     "and set from_papers to true.\n"
     "- If they don't, answer from your general knowledge and set from_papers to false.\n"
     "Never present general knowledge as coming from the papers."),
    ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
])


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm       = ChatOpenAI(model=MODEL_NAME, temperature=0).with_structured_output(RagAnswer)
    chain     = RAG_PROMPT | llm

    def run(inputs: dict) -> dict:
        question = inputs["query"]
        docs     = retriever.invoke(question)
        context  = "\n\n---\n\n".join(d.page_content for d in docs)
        output   = chain.invoke({"context": context, "question": question})
        return {
            "result":           output.answer,
            "from_papers":      output.from_papers,
            "source_documents": docs,
        }

    # same input/output keys as the old RetrievalQA chain
    return RunnableLambda(run)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force reprocess all PDFs")
    parser.add_argument("--query",   type=str, default=None)
    args = parser.parse_args()

    # load or build vectorstore
    if vectorstore_exists() and not args.rebuild:
        print("Vectorstore found — loading saved index.")
        print("Use --rebuild to reprocess all papers.\n")
        vectorstore = load_vectorstore()
    else:
        all_chunks  = process_all_papers(PAPERS_DIR)
        print("\nBuilding vectorstore")
        vectorstore = build_and_save_vectorstore(all_chunks)

    # query
    query = args.query or "What is the role of coral reefs?"
    chain = build_rag_chain(vectorstore)

    print(f"\nQuery: {query}")
    response = chain.invoke({"query": query})

    print("\n--- Answer ---")
    print(response["result"])
    print(f"\n(from papers: {response['from_papers']})")

    print("\n--- Source Chunks ---")
    for i, doc in enumerate(response["source_documents"], 1):
        print(f"\n[{i}] {doc.page_content[:500]}")