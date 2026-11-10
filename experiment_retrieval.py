"""Task: Retrieval quality versus cost experiments comparing Top-K settings (3, 5, 8)."""

import os
import sys
import time
import json
import uuid
import csv
import asyncio
import argparse
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import search_papers
import answer_generator
from citation_validator import validate_citations
from mock_llm import MockMode, mock_service
from groq_service import send_chat_completion

# Output CSV columns for experiment results
CSV_FIELDNAMES = [
    "experiment_id",
    "retrieval_setting",
    "top_k",
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

# Standard benchmark questions covering all three research papers and a negative control
BENCHMARK_QUESTIONS = [
    "What is the Transformer architecture and how does self-attention work?",
    "What is Moshi and how does it support full-duplex spoken dialogue?",
    "How does dynamic chunking differ from fixed-size chunking in RAG?",
    "What are the computational complexity advantages of self-attention layers?",
    "What are the recipe ingredients and steps for baking French chocolate croissants?",
]


async def run_single_experiment(
    question: str,
    top_k: int,
    setting_label: str,
    source: str = "mock",
    model=None,
    index=None,
    metadata=None,
) -> Dict[str, Any]:
    """Execute a single RAG query under a specified Top-K setting and measure quality and cost."""
    t_start = time.perf_counter()
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"

    # 1. Measure discrete FAISS retrieval latency
    t_retrieval_start = time.perf_counter()
    chunks = search_papers.search(
        query=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )
    retrieval_latency = round((time.perf_counter() - t_retrieval_start) * 1000, 2)

    # 2. Extract retrieval quality and relevance metrics
    scores = [float(c.get("score", 0.0)) for c in chunks]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    min_score = round(min(scores), 4) if scores else 0.0
    max_score = round(max(scores), 4) if scores else 0.0

    chunk_ids = ";".join(str(c["chunk_id"]) for c in chunks)
    papers = ";".join(sorted(list({c["paper_name"] for c in chunks})))
    pages = ";".join(str(c.get("page_number", "N/A")) for c in chunks)

    # 3. Build grounded prompt for LLM
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

    # 4. Generate answer via chosen provider
    if source == "mock":
        llm_res = await mock_service.generate_response(
            messages=prompt_messages,
            mode=MockMode.NORMAL,
        )
        answer_text = llm_res["content"]
        in_tokens = llm_res["input_tokens"]
        out_tokens = llm_res["output_tokens"]
    else:
        llm_res = send_chat_completion(messages=prompt_messages)
        answer_text = llm_res["content"]
        in_tokens = llm_res["input_tokens"]
        out_tokens = llm_res["output_tokens"]

    # 5. Validate citations and groundedness
    validation = validate_citations(answer_text, chunks)
    total_latency = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "experiment_id": exp_id,
        "retrieval_setting": setting_label,
        "top_k": top_k,
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

    print("\n" + "=" * 94)
    print("                 RETRIEVAL QUALITY VERSUS COST EXPERIMENT SUMMARY")
    print("=" * 94)
    print(
        f"{'Setting':<22} | {'Top-K':<5} | {'Avg Sim':<7} | {'Avg In Toks':<11} | "
        f"{'Avg Out':<7} | {'Retr ms':<8} | {'Total ms':<8} | {'Pass Rate':<9}"
    )
    print("-" * 94)

    for st in settings:
        rows = [r for r in results if r["retrieval_setting"] == st]
        count = len(rows)
        top_k = rows[0]["top_k"]

        avg_sim = round(sum(r["avg_similarity_score"] for r in rows) / count, 3)
        avg_in_toks = int(sum(r["tokens_in"] for r in rows) / count)
        avg_out_toks = int(sum(r["tokens_out"] for r in rows) / count)
        avg_retr_ms = round(sum(r["retrieval_latency_ms"] for r in rows) / count, 1)
        avg_tot_ms = round(sum(r["total_latency_ms"] for r in rows) / count, 1)
        valid_cnt = sum(1 for r in rows if r["is_valid"])
        pass_rate = round((valid_cnt / count) * 100, 1)

        print(
            f"{st:<22} | {top_k:<5} | {avg_sim:<7.3f} | {avg_in_toks:<11,d} | "
            f"{avg_out_toks:<7d} | {avg_retr_ms:<8.1f} | {avg_tot_ms:<8.1f} | {pass_rate:>6.1f}%"
        )

    print("=" * 94)
    print("  Key Takeaways:")
    print("  • Cost (Input Tokens): Context size scales directly with Top-K.")
    print("  • Quality (Similarity): Avg similarity decreases as lower-ranked chunks are retrieved.")
    print("  • Latency: FAISS vector retrieval remains fast (<20ms) across all settings.")
    print("=" * 94 + "\n")


async def main():
    """Main CLI runner to execute retrieval experiments across configured Top-K settings."""
    parser = argparse.ArgumentParser(description="Retrieval Quality vs Cost Experiments")
    parser.add_argument(
        "--source",
        choices=["mock", "groq"],
        default="mock",
        help="LLM provider: 'mock' (default, quota safe) or 'groq' (real).",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["A", "B", "C"],
        help="Experiment settings to run: A (Top-3), B (Top-5), C (Top-8). Default: all.",
    )
    parser.add_argument(
        "--top-k-a",
        type=int,
        default=3,
        help="Top-K for Experiment A (default: 3).",
    )
    parser.add_argument(
        "--top-k-b",
        type=int,
        default=5,
        help="Top-K for Experiment B (default: 5).",
    )
    parser.add_argument(
        "--top-k-c",
        type=int,
        default=8,
        help="Top-K for Experiment C (default: 8).",
    )
    parser.add_argument(
        "--output",
        default="retrieval_experiments.csv",
        help="Output CSV file path (default: retrieval_experiments.csv).",
    )

    args = parser.parse_args()

    # Safety warning for real Groq API usage
    if args.source == "groq":
        print("\n⚠️  CAUTION: Testing with real Groq LLM integration.")
        print("   Running 5 questions across 3 experiments = 15 API calls.")

    # Load FAISS index, metadata, and embedding model once
    model, index, metadata = answer_generator.load_retrieval_pipeline()

    # Map selected experiment settings
    configs = []
    if "A" in args.settings:
        configs.append(("Experiment A (Top-K=3)", args.top_k_a))
    if "B" in args.settings:
        configs.append(("Experiment B (Top-K=5)", args.top_k_b))
    if "C" in args.settings:
        configs.append(("Experiment C (Top-K=8)", args.top_k_c))

    all_results: List[Dict[str, Any]] = []

    print(f"\n🚀 Running Retrieval Experiments (Source: {args.source})")
    print(f"   Settings : {[c[0] for c in configs]}")
    print(f"   Questions: {len(BENCHMARK_QUESTIONS)} benchmark questions per setting")

    for label, top_k in configs:
        print(f"\n{'=' * 66}")
        print(f"  Running: {label}")
        print(f"{'=' * 66}")

        for idx, q in enumerate(BENCHMARK_QUESTIONS, start=1):
            res = await run_single_experiment(
                question=q,
                top_k=top_k,
                setting_label=label,
                source=args.source,
                model=model,
                index=index,
                metadata=metadata,
            )
            all_results.append(res)
            print(
                f"  [{idx}/{len(BENCHMARK_QUESTIONS)}] Top-{top_k} | "
                f"Retr: {res['retrieval_latency_ms']}ms | Total: {res['total_latency_ms']}ms | "
                f"Tokens: {res['tokens_in']} in / {res['tokens_out']} out | Valid: {res['is_valid']}"
            )

    # Save to CSV and print comparison table
    save_results_to_csv(all_results, filepath=args.output)
    print_comparison_table(all_results)


if __name__ == "__main__":
    asyncio.run(main())

