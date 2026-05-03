
# PDF processing, chunking, titling, vectorstore
import os
import re
import sys
import tiktoken
import pypdf
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(".."))
from helper_functions import replace_t_with_space, EmbeddingProvider, get_langchain_embedding_provider

from config import (
    MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_WORKERS, SCIENTIFIC_SEPARATORS, VECTORSTORE_DIR,
)

TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")


# tokenizer helpers 

def token_length(text: str) -> int:
    return len(TOKEN_ENCODER.encode(text, disallowed_special=()))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = TOKEN_ENCODER.encode(text, disallowed_special=())
    return TOKEN_ENCODER.decode(tokens[:max_tokens])


# article cleanup 

def remove_references_section(text: str) -> str:
    # cut at a known section heading
    heading_pattern = re.compile(
        r'[\s\d]*('
        r'References and Notes|References|Bibliography|Works Cited'
        r'|Literature Cited|Supplementary Materials?|Supplementary Information'
        r'|Supporting Information|Acknowledgements?|Competing interests'
        r'|Author contributions|Data availability|Code availability'
        r'|© The Author'
        r')\b.*',
        re.IGNORECASE | re.DOTALL,
    )
    cut = heading_pattern.sub("", text).strip()
    if len(cut) < len(text) * 0.95:
        return cut

    # find first reference entry pattern in last 40% of document
    # handles: "[1] Author" or "1. Author" styles
    cutoff = int(len(text) * 0.60)
    tail   = text[cutoff:]
    match  = re.search(
        r'\n(\[\d+\]|\d+\.)\s+[A-Z][a-zA-Z\-]+,?\s+[A-Z].*',
        tail, re.DOTALL,
    )
    if match:
        return text[:cutoff + match.start()].strip()

    return text


def load_and_clean_pdf(file_path: str) -> str:
    pages = []
    reader = pypdf.PdfReader(file_path)
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)

    text = re.sub(r"arXiv:.*?\n", "", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*[\dIVXivx]+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r'(https?://)(\s*)([^\s]+(\.\s*[^\s]+)*)', lambda m: '', text)
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.com\S*', '', text)
    text = remove_references_section(text)

    return text.strip()

# LLM calls

def make_llm_call(messages: list[dict], max_tokens: int) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=max_tokens, temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def get_document_title(document_text: str) -> str:
    prompt = (
        "What is the title of the following document?\n"
        "Respond with the title only. Nothing else.\n\n"
        f"DOCUMENT:\n{truncate_to_tokens(document_text, 4000)}"
    )
    return make_llm_call([{"role": "user", "content": prompt}], max_tokens=32)


def get_chunk_title(chunk_text: str, document_title: str) -> str:
    prompt = (
        f'You are reading an excerpt from a scientific paper titled: "{document_title}"\n\n'
        "Write a short descriptive title (5-10 words) for THIS specific excerpt.\n"
        "Respond with the title only. No punctuation at the end.\n\n"
        f"EXCERPT:\n{truncate_to_tokens(chunk_text, 600)}"
    )
    return make_llm_call([{"role": "user", "content": prompt}], max_tokens=20)


# chunking + parallel titling

def split_into_chunks(text: str, document_title: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        separators=SCIENTIFIC_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_length,
        is_separator_regex=False,
    )
    docs      = splitter.create_documents([text])
    docs      = replace_t_with_space(docs)
    raw_texts = [d.page_content.strip() for d in docs if d.page_content.strip()]

    print(f"  Generating titles for {len(raw_texts)} chunks (parallel)...")
    titles = [""] * len(raw_texts)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_chunk_title, t, document_title): i
            for i, t in enumerate(raw_texts)
        }
        for future in as_completed(futures):
            i = futures[future]
            titles[i] = future.result()
            print(f"✓ chunk {i+1}/{len(raw_texts)}: {titles[i]}")

    return [
        {"text": t, "title": title, "document_title": document_title}
        for t, title in zip(raw_texts, titles)
    ]


# vectorstore

def build_and_save_vectorstore(all_chunks: list[dict]) -> FAISS:
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    texts = [
        f"Paper: {c['document_title']}\nSection: {c['title']}\n\n{c['text']}"
        for c in all_chunks
    ]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vectorstore saved → {VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    return FAISS.load_local(
        VECTORSTORE_DIR, embeddings,
        allow_dangerous_deserialization=True,
    )


def vectorstore_exists() -> bool:
    return os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss"))