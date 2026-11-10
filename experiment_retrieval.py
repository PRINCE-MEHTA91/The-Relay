"""Task: Retrieval quality versus cost experiments comparing three genuinely distinct configurations."""

import os
import sys
import time
import json
import uuid
import csv
import asyncio
import argparse
import numpy as np
import faiss
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import search_papers
import answer_generator
import chunk_papers
from citation_validator import validate_citations
from mock_llm import MockMode, mock_service
from groq_service import send_chat_completion

PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
CHUNKS_250_FILE = os.path.join(PROCESSED_DIR, "chunks_250.json")
INDEX_250_FILE = os.path.join(PROCESSED_DIR, "faiss_index_250.bin")

# Output CSV columns for retrieval experiments
CSV_FIELDNAMES = [
    "experiment_id",
    "retrieval_setting",
    "chunk_size",
    "top_k",
    "reranking",
    "source",
    "question",
    "retrieval_latency_ms",
    "total_latency_ms",
    "tokens_in",
    "tokens_out",
    "total_tokens",
    "avg_similarity_score",
    "min_similarity_score",
    "max_similarity_score",
    "retrieved_chunk_ids",
    "retrieved_papers",
    "retrieved_pages",
    "is_valid",
    "validation_status",
    "citations_count",
    "unsupported_claims_count",
    "answer_preview",
]

# Standard benchmark questions covering all three papers and a negative control
BENCHMARK_QUESTIONS = [
    "What is the Transformer architecture and how does self-attention work?",
    "What is Moshi and how does it support full-duplex spoken dialogue?",
    "How does dynamic chunking differ from fixed-size chunking in RAG?",
    "What are the computational complexity advantages of self-attention layers?",
    "What are the recipe ingredients and steps for baking French chocolate croissants?",
]


def get_or_build_250_token_pipeline(model) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load or generate 250-token chunks and corresponding FAISS index."""
    if os.path.exists(CHUNKS_250_FILE) and os.path.exists(INDEX_250_FILE):
        with open(CHUNKS_250_FILE, "r", encoding="utf-8") as f:
            metadata_250 = json.load(f)
        index_250 = faiss.read_index(INDEX_250_FILE)
        return index_250, metadata_250

    print("⚡ Generating 250-token chunks for Setting A...")
    chunks_250 = chunk_papers.build_chunks_for_size(chunk_size=250, overlap=25)
    texts = [c["text"] for c in chunks_250]

    print(f"🤖 Embedding {len(texts)} chunks of 250 tokens...")
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    index_250 = faiss.IndexFlatIP(embeddings.shape[1])
    index_250.add(embeddings)

    with open(CHUNKS_250_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_250, f, indent=2)
    faiss.write_index(index_250, INDEX_250_FILE)

    return index_250, chunks_250


async def run_single_retrieval_query(
    question: str,
    setting_name: str,
    chunk_size: int,
    top_k: int,
    reranking: bool,
    source: str,
    model,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a single query under specified retrieval setting and record metrics."""
    t_start = time.perf_counter()
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"

    # 1. Retrieval
    t_retrieval_start = time.perf_counter()
    if reranking:
        # Retrieve candidate pool then re-rank with hybrid lexical + semantic scoring
        candidates = search_papers.search(query=question, model=model, index=index, metadata=metadata, top_k=top_k * 2)
        chunks = search_papers.hybrid_rerank(query=question, candidates=candidates, top_k=top_k)
    else:
        chunks = search_papers.search(query=question, model=model, index=index, metadata=metadata, top_k=top_k)
    retrieval_latency = round((time.perf_counter() - t_retrieval_start) * 1000, 2)

    # 2. Extract relevance stats
    scores = [float(c.get("score", 0.0)) for c in chunks]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    min_score = round(min(scores), 4) if scores else 0.0
    max_score = round(max(scores), 4) if scores else 0.0

    chunk_ids = ";".join(str(c["chunk_id"]) for c in chunks)
    papers = ";".join(sorted(list({c["paper_name"] for c in chunks})))
    pages = ";".join(str(c.get("page_number", "N/A")) for c in chunks)

    # 3. Construct prompt
    context_text = answer_generator.format_context_passages(chunks)
    system_prompt = (
        "You are an expert research assistant for 'The Relay'.\n"
        "Your task is to answer technical questions strictly using only the provided research paper passages.\n\n"
        "Rules:\n"
        "1. Base your answer ONLY on the provided passages. Do not use pre-trained or external knowledge.\n"
        "2. If the passages do not contain enough information to answer the question, state exactly:\n"
        "   'The answer cannot be found in the provided research papers.'\n"
        "3. Include citations in brackets for every claim (e.g., [Attention Is All You Need, p. 3])."
    )
    user_prompt = (
        f"Context Passages:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer clearly and concisely with citations:"
    )
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 4. Generate answer
    if source == "stand-in":
        llm_res = await mock_service.generate_response(messages=prompt_messages, mode=MockMode.NORMAL)
        answer_text = llm_res["content"]
        in_tokens = llm_res["input_tokens"]
        out_tokens = llm_res["output_tokens"]
    else:
        llm_res = send_chat_completion(messages=prompt_messages)
        answer_text = llm_res["content"]
        in_tokens = llm_res["input_tokens"]
        out_tokens = llm_res["output_tokens"]

    # 5. Citation validation
    validation = validate_citations(answer_text, chunks)
    total_latency = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "experiment_id": exp_id,
        "retrieval_setting": setting_name,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "reranking": "Enabled" if reranking else "Disabled",
        "source": source,
        "question": question,
        "retrieval_latency_ms": retrieval_latency,
        "total_latency_ms": total_latency,
        "tokens_in": in_tokens,
        "tokens_out": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "avg_similarity_score": avg_score,
        "min_similarity_score": min_score,
        "max_similarity_score": max_score,
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_papers": papers,
        "retrieved_pages": pages,
        "is_valid": validation["is_valid"],
        "validation_status": validation["validation_status"],
        "citations_count": len(validation["validated_citations"]),
        "unsupported_claims_count": len(validation["unsupported_claims"]),
        "answer_preview": (answer_text[:120] + "...") if len(answer_text) > 120 else answer_text,
    }


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str = "retrieval_experiments.csv"):
    """Write experiment outputs to CSV file."""
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})
    print(f"\n📁 Saved {len(results)} experiment records to {filepath}")


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print an aggregated trade-off table comparing quality versus token cost and latency."""
    settings = sorted(list({r["retrieval_setting"] for r in results}))
    if not settings:
        return

    print("\n" + "=" * 105)
    print("                 RETRIEVAL QUALITY VERSUS COST EXPERIMENT SUMMARY")
    print("=" * 105)
    print(
        f"{'Setting':<25} | {'Size':<6} | {'Top-K':<5} | {'Rerank':<8} | {'Avg Sim':<7} | "
        f"{'Avg In Toks':<11} | {'Retr ms':<8} | {'Total ms':<8} | {'Pass Rate':<9}"
    )
    print("-" * 105)

    for st in settings:
        rows = [r for r in results if r["retrieval_setting"] == st]
        count = len(rows)
        chunk_sz = rows[0]["chunk_size"]
        top_k = rows[0]["top_k"]
        rerank = rows[0]["reranking"]

        avg_sim = round(sum(r["avg_similarity_score"] for r in rows) / count, 3)
        avg_in_toks = int(sum(r["tokens_in"] for r in rows) / count)
        avg_retr_ms = round(sum(r["retrieval_latency_ms"] for r in rows) / count, 1)
        avg_tot_ms = round(sum(r["total_latency_ms"] for r in rows) / count, 1)
        valid_cnt = sum(1 for r in rows if r["is_valid"])
        pass_rate = round((valid_cnt / count) * 100, 1)

        print(
            f"{st:<25} | {chunk_sz:<6} | {top_k:<5} | {rerank:<8} | {avg_sim:<7.3f} | "
            f"{avg_in_toks:<11,d} | {avg_retr_ms:<8.1f} | {avg_tot_ms:<8.1f} | {pass_rate:>6.1f}%"
        )

    print("=" * 105)


async def main():
    """Main CLI runner for the three retrieval quality versus cost configurations."""
    parser = argparse.ArgumentParser(description="Retrieval Quality vs Cost Experiments")
    parser.add_argument(
        "--source",
        choices=["stand-in", "real"],
        default="stand-in",
        help="Provider: 'stand-in' (default) or 'real'.",
    )
    parser.add_argument(
        "--output",
        default="retrieval_experiments.csv",
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    model, index_500, metadata_500 = answer_generator.load_retrieval_pipeline()
    index_250, metadata_250 = get_or_build_250_token_pipeline(model)

    configs = [
        ("Setting A (250t, Top-6, No-Rerank)", 250, 6, False, index_250, metadata_250),
        ("Setting B (500t, Top-4, No-Rerank)", 500, 4, False, index_500, metadata_500),
        ("Setting C (500t, Top-8, Hybrid-Rerank)", 500, 8, True, index_500, metadata_500),
    ]

    all_results: List[Dict[str, Any]] = []

    for label, chunk_sz, top_k, rerank, idx_obj, meta_obj in configs:
        print(f"\n{'=' * 66}\n  Running: {label}\n{'=' * 66}")
        for q in BENCHMARK_QUESTIONS:
            res = await run_single_retrieval_query(
                question=q,
                setting_name=label,
                chunk_size=chunk_sz,
                top_k=top_k,
                reranking=rerank,
                source=args.source,
                model=model,
                index=idx_obj,
                metadata=meta_obj,
            )
            all_results.append(res)
            print(f"  • Q: {q[:45]}... -> Status: {res['validation_status']}, Tokens: {res['tokens_in']}, Sim: {res['avg_similarity_score']}")

    save_results_to_csv(all_results, filepath=args.output)
    print_comparison_table(all_results)


if __name__ == "__main__":
    asyncio.run(main())
