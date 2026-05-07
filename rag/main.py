
import os
import sys
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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
        import regex as re
        for match in re.finditer(r'.{100}[Rr]eferences.{100}', text):
            print(repr(match.group()))
            print("---")
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


def build_rag_chain(vectorstore):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Answer the question based only on the following context.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question: {question}

        Answer:
        """,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=MODEL_NAME, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


if __name__ == "__main__":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

    print("\n--- Source Chunks ---")
    for i, doc in enumerate(response["source_documents"], 1):
        print(f"\n[{i}] {doc.page_content[:500]}")