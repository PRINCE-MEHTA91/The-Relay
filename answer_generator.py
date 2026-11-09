"""Task 9: Research question answering using FAISS retrieval and Groq LLM."""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from groq_service import send_chat_completion, get_groq_client
import search_papers
from citation_validator import validate_citations

# Config and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
INDEX_FILE = os.path.join(PROCESSED_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(PROCESSED_DIR, "metadata.json")
ALT_METADATA_FILE = os.path.join(PROCESSED_DIR, "chunk_metadata.json")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# Global cache for retrieval assets to prevent redundant disk/model reloads
_RETRIEVAL_CACHE: Dict[str, Any] = {}


def load_retrieval_pipeline():
    """Load and cache the FAISS index, metadata records, and embedding model."""
    if _RETRIEVAL_CACHE:
        return _RETRIEVAL_CACHE["model"], _RETRIEVAL_CACHE["index"], _RETRIEVAL_CACHE["metadata"]

    meta_path = METADATA_FILE if os.path.exists(METADATA_FILE) else ALT_METADATA_FILE
    if not os.path.exists(INDEX_FILE) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"FAISS index or metadata missing. Run generate_embeddings.py and search_papers.py first."
        )

    print("📦 Loading FAISS index and metadata...")
    index = faiss.read_index(INDEX_FILE)
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
        # Check if the paper name is cited in text or if the chunk is top relevant
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
    """
    Complete RAG workflow:
    1. Retrieve top passages with FAISS
    2. Build context-grounded prompt
    3. Query Groq LLM with retries
    4. Return structured response with citations and metrics
    """
    overall_start = time.perf_counter()

    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Question cannot be empty.",
            "citations": [],
            "retrieved_chunks": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
            "status": "error",
        }

    # 1. Retrieve relevant passages
    model, index, metadata = load_retrieval_pipeline()
    retrieved_chunks = search_papers.search(
        query=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "The answer cannot be found in the provided research papers.",
            "citations": [],
            "retrieved_chunks": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            "status": "success",
        }

    # 2. Build system and user prompt with grounding constraints
    context_text = format_context_passages(retrieved_chunks)

    system_prompt = (
        "You are an expert research assistant for 'The Relay'.\n"
        "Your task is to answer technical questions strictly using only the provided research paper passages.\n\n"
        "Rules:\n"
        "1. Base your answer ONLY on the provided passages. Do not use pre-trained or external knowledge.\n"
        "2. If the passages do not contain enough information to answer the question, state exactly:\n"
        "   'The answer cannot be found in the provided research papers.'\n"
        "3. Include citations in brackets for every claim (e.g., [Attention Is All You Need, p. 3]).\n"
        "4. Do not invent citations or extrapolate facts not present in the excerpts."
    )

    user_prompt = (
        f"Context Passages:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer clearly and concisely with citations:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 3. Query Groq API
    groq_res = send_chat_completion(
        messages=messages,
        model=model_override,
    )

    total_latency = round((time.perf_counter() - overall_start) * 1000, 2)
    answer_text = groq_res["content"].strip()

    # 4. Citation and claim validation (Task 10)
    if groq_res["status"] == "success":
        validation = validate_citations(
            answer=answer_text,
            retrieved_chunks=retrieved_chunks,
        )
        final_answer = answer_text if validation["is_valid"] else f"Unverified Answer: {answer_text}"
        status = "verified" if validation["is_valid"] else "unverified"
    else:
        validation = {
            "is_valid": False,
            "citation_errors": [groq_res.get("error", "Unknown error")],
            "unsupported_claims": [],
            "validated_citations": [],
            "validation_status": "GENERATION_ERROR",
        }
        final_answer = f"Generation error: {groq_res.get('error', 'Unknown error')}"
        status = "error"

    return {
        "question": question,
        "answer": final_answer,
        "raw_answer": answer_text,
        "is_valid": validation["is_valid"],
        "validation": validation,
        "citations": validation["validated_citations"],
        "retrieved_chunks": retrieved_chunks,
        "input_tokens": groq_res["input_tokens"],
        "output_tokens": groq_res["output_tokens"],
        "latency_ms": total_latency,
        "status": status,
    }


def run_test_suite():
    """Run verification tests on research questions and unsupported query."""
    print("=" * 66)
    print("  THE RELAY — Task 9: AI Answer Generation Test Suite")
    print("=" * 66)

    test_questions = [
        "What is the Transformer architecture?",
        "How does self-attention work?",
        "What is Moshi?",
        "Explain dynamic chunking.",
        "What are the recipe ingredients and steps for baking French chocolate croissants?",
    ]

    for idx, q in enumerate(test_questions, start=1):
        print(f"\n{'=' * 66}")
        print(f"  Test {idx}/5: {q}")
        print(f"{'=' * 66}")

        result = answer_question(q)

        print(f"Status     : {result['status']} (Verified: {result['is_valid']})")
        print(f"Validation : {result['validation']['validation_status']}")
        print(f"Latency    : {result['latency_ms']} ms")
        print(f"Tokens     : {result['input_tokens']} in / {result['output_tokens']} out")
        print(f"\n--- Answer ---")
        print(result["answer"])

        print(f"\n--- Citations ({len(result['citations'])} sources) ---")
        if result["citations"]:
            for c in result["citations"][:2]:  # Show top 2 citations
                print(f" • [{c['paper_name']}, p. {c['page_number']}] (Chunk #{c['chunk_id']})")
                print(f"   Excerpt: \"{c['source_passage'][:120]}...\"")
        else:
            print(" • No citations (question unsupported or refused as expected)")


if __name__ == "__main__":
    run_test_suite()

