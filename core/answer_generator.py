"""Research question answering using FAISS retrieval and Groq LLM."""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Setup project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe imports supporting both modular and root-level execution
try:
    from core.groq_service import send_chat_completion, get_groq_client
except ImportError:
    from groq_service import send_chat_completion, get_groq_client

try:
    from rag import search_papers
except ImportError:
    import search_papers

try:
    from rag.citation_validator import validate_citations
except ImportError:
    from citation_validator import validate_citations

# Config and file paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
FALLBACK_PROCESSED = PROJECT_ROOT / "processed"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


def get_index_path() -> Path:
    for p in [DATA_EMBEDDINGS / "faiss_index.bin", DATA_PROCESSED / "faiss_index.bin", FALLBACK_PROCESSED / "faiss_index.bin"]:
        if p.exists():
            return p
    return DATA_EMBEDDINGS / "faiss_index.bin"


def get_metadata_path() -> Path:
    for p in [DATA_PROCESSED / "metadata.json", FALLBACK_PROCESSED / "metadata.json", DATA_PROCESSED / "chunk_metadata.json", FALLBACK_PROCESSED / "chunk_metadata.json"]:
        if p.exists():
            return p
    return DATA_PROCESSED / "metadata.json"


INDEX_FILE = str(get_index_path())
METADATA_FILE = str(get_metadata_path())
ALT_METADATA_FILE = str(FALLBACK_PROCESSED / "chunk_metadata.json")

# Global cache for retrieval assets to prevent redundant disk/model reloads
_RETRIEVAL_CACHE: Dict[str, Any] = {}


def load_retrieval_pipeline():
    """Load and cache the FAISS index, metadata records, and embedding model."""
    if _RETRIEVAL_CACHE:
        return _RETRIEVAL_CACHE["model"], _RETRIEVAL_CACHE["index"], _RETRIEVAL_CACHE["metadata"]

    idx_path = get_index_path()
    meta_path = get_metadata_path()

    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"FAISS index ({idx_path}) or metadata ({meta_path}) missing. Run generate_embeddings.py and search_papers.py first."
        )

    print(f"📦 Loading FAISS index from {idx_path}...")
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"🤖 Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    _RETRIEVAL_CACHE["index"] = index
    _RETRIEVAL_CACHE["metadata"] = metadata
    _RETRIEVAL_CACHE["model"] = model
    return model, index, metadata


def format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved passages into a structured context string for the prompt."""
    formatted_blocks = []
    for i, c in enumerate(chunks, start=1):
        block = (
            f"[Passage {i}]\n"
            f"Paper: {c['paper_name']}\n"
            f"Page: {c.get('page_number', 'N/A')}\n"
            f"Chunk ID: {c['chunk_id']}\n"
            f"Content:\n{c['text']}\n"
        )
        formatted_blocks.append(block)
    return "\n----------------------------------------\n".join(formatted_blocks)


def extract_citations(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify which retrieved passages were referenced or relevant to the answer."""
    unsupported_phrases = [
        "cannot be found",
        "not found in the provided",
        "no information",
        "does not mention",
    ]
    if any(phrase in answer.lower() for phrase in unsupported_phrases):
        return []

    citations = []
    for c in chunks:
        citations.append({
            "paper_name": c["paper_name"],
            "page_number": c.get("page_number", "N/A"),
            "chunk_id": c["chunk_id"],
            "similarity_score": c.get("score", 0.0),
            "source_passage": c["text"][:250].strip() + "..." if len(c["text"]) > 250 else c["text"].strip(),
        })
    return citations


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve top passages with FAISS, query LLM, and validate citations."""
    overall_start = time.perf_counter()

    # 1. Load retrieval assets
    retrieval_start = time.perf_counter()
    model, index, metadata = load_retrieval_pipeline()

    # 2. Vector search
    retrieved_chunks = search_papers.search(
        query=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )
    retrieval_time_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "No relevant research paper passages could be found.",
            "is_valid": False,
            "validation": {
                "is_valid": False,
                "citation_errors": ["No context chunks retrieved."],
                "unsupported_claims": [],
                "validated_citations": [],
                "validation_status": "REJECTED_NO_CONTEXT",
            },
            "citations": [],
            "retrieved_chunks": [],
            "retrieval_latency_ms": retrieval_time_ms,
            "latency_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retries_attempted": 0,
            "status": "error",
        }

    # 3. Assemble prompt
    context_text = format_context_passages(retrieved_chunks)
    system_prompt = (
        "You are an expert AI research assistant. Your task is to answer the user's question "
        "using ONLY the retrieved research paper passages below. "
        "Strict rules:\n"
        "1. Answer ONLY based on the facts directly stated in the context.\n"
        "2. If the context does not contain sufficient facts to answer the question, state: "
        "'The answer cannot be found in the provided research papers.' Do not guess.\n"
        "3. Every factual statement must cite its source passage in brackets: [Paper Name, p. X] or [Paper Name].\n"
        "4. Be concise, technical, and mathematically precise."
    )
    user_prompt = f"Context passages:\n{context_text}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 4. Invoke LLM via Groq Service
    llm_response = send_chat_completion(
        messages=messages,
        model=model_override,
    )

    if llm_response["status"] != "success":
        return {
            "question": question,
            "answer": f"Error from LLM provider: {llm_response.get('error', 'Unknown error')}",
            "is_valid": False,
            "validation": {
                "is_valid": False,
                "citation_errors": [llm_response.get("error", "LLM failure")],
                "unsupported_claims": [],
                "validated_citations": [],
                "validation_status": "LLM_ERROR",
            },
            "citations": [],
            "retrieved_chunks": retrieved_chunks,
            "retrieval_latency_ms": retrieval_time_ms,
            "latency_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            "input_tokens": llm_response.get("input_tokens", 0),
            "output_tokens": llm_response.get("output_tokens", 0),
            "total_tokens": llm_response.get("total_tokens", 0),
            "retries_attempted": llm_response.get("retries_attempted", 0),
            "status": "error",
        }

    raw_answer = llm_response["content"].strip()

    # 5. Citation and Groundedness Validation
    validation_start = time.perf_counter()
    validation_results = validate_citations(raw_answer, retrieved_chunks)
    val_time_ms = round((time.perf_counter() - validation_start) * 1000, 2)

    total_latency_ms = round((time.perf_counter() - overall_start) * 1000, 2)

    return {
        "question": question,
        "answer": raw_answer,
        "is_valid": validation_results["is_valid"],
        "validation": validation_results,
        "citations": validation_results["validated_citations"],
        "retrieved_chunks": retrieved_chunks,
        "retrieval_latency_ms": retrieval_time_ms,
        "validation_latency_ms": val_time_ms,
        "latency_ms": total_latency_ms,
        "input_tokens": llm_response["input_tokens"],
        "output_tokens": llm_response["output_tokens"],
        "total_tokens": llm_response["total_tokens"],
        "retries_attempted": llm_response["retries_attempted"],
        "model": llm_response["model"],
        "status": "verified" if validation_results["is_valid"] else "unverified",
    }


def main():
    """CLI runner for interactive research questions."""
    import argparse
    parser = argparse.ArgumentParser(description="Query research papers with FAISS RAG pipeline.")
    parser.add_argument("question", nargs="?", help="Research question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    args = parser.parse_args()

    q = args.question or "What is the mathematical formulation of Scaled Dot-Product Attention in the Transformer?"
    print(f"\n🔬 Query: {q}\n")
    res = answer_question(q, top_k=args.top_k)

    print("=" * 66)
    print(f"STATUS  : {res['status'].upper()}")
    print(f"VALID   : {res['is_valid']}")
    print(f"LATENCY : {res['latency_ms']} ms (retrieval: {res['retrieval_latency_ms']} ms)")
    print(f"TOKENS  : {res['input_tokens']} in / {res['output_tokens']} out")
    print("-" * 66)
    print("ANSWER:")
    print(res["answer"])
    print("-" * 66)
    print("CITATIONS:")
    for c in res["citations"]:
        ver = "✅" if c.get("verified") else "⚠️"
        print(f"  {ver} [{c['paper_name']}, p. {c['page_number']}] (chunk #{c['chunk_id']})")
    print("=" * 66)


if __name__ == "__main__":
    main()

