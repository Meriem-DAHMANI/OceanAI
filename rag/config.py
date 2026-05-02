PAPERS_DIR      = "oceanAI/rag/research_papers"
VECTORSTORE_DIR = "oceanAI/rag/vectorstore"
 
MODEL_NAME    = "gpt-4o-mini"
CHUNK_SIZE    = 512   # tokens
CHUNK_OVERLAP = 100   # tokens (~20%)
TOP_K         = 3     # chunks retrieved per query
MAX_WORKERS   = 10    # parallel LLM calls for chunk titling
 
SCIENTIFIC_SEPARATORS = [
    "\n\nAbstract", "\n\nIntroduction", "\n\nBackground",
    "\n\nLiterature Review", "\n\nMethods", "\n\nMethodology",
    "\n\nMaterials and Methods", "\n\nResults", "\n\nDiscussion",
    "\n\nConclusion", "\n\nConclusions", "\n\nAcknowledg",
    "\n\nAppendix", "\n\n", "\n", ". ", " ", "",
]
