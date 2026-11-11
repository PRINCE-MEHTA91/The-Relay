"""Answer quality evaluation system grading RAG responses against 15 verified ground-truth questions."""

import os
import sys
import time
import json
import uuid
import csv
import asyncio
import argparse
from typing import Dict, Any, List, Optional, Tuple

from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from core import answer_generator
    from core.mock_llm import MockMode, mock_service
    from core.groq_service import send_chat_completion
    from rag import search_papers
    from rag.citation_validator import (
        validate_citations,
        parse_citations,
        normalize_title,
        matches_paper_name,
        is_valid_refusal,
    )
except ImportError:
    import answer_generator
    import search_papers
    from citation_validator import (
        validate_citations,
        parse_citations,
        normalize_title,
        matches_paper_name,
        is_valid_refusal,
    )
    from mock_llm import MockMode, mock_service
    from groq_service import send_chat_completion

# Output CSV columns for answer evaluations
CSV_FIELDNAMES = [
    "question_id",
    "question_type",
    "question",
    "expected_answer",
    "expected_paper",
    "expected_page",
    "actual_answer",
    "cited_paper",
    "cited_page",
    "citation_paper_correct",
    "citation_page_correct",
    "answer_grade",
    "retrieval_setting",
    "retrieval_latency_ms",
    "total_latency_ms",
    "tokens_in",
    "tokens_out",
    "is_valid",
    "validation_status",
]

# 15 ground-truth questions verified directly from the three arXiv paper PDFs
EVALUATION_DATASET: List[Dict[str, Any]] = [
    # --- Attention Is All You Need (1706.03762 v7) ---
    {
        "id": "Q01",
        "type": "answerable",
        "paper": "Attention Is All You Need",
        "page": 3,
        "question": "What overall architecture does the Transformer use?",
        "expected_answer": "An encoder-decoder architecture using stacked self-attention and point-wise fully connected feed-forward layers.",
        "key_evidence": "The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.",
        "key_terms": ["encoder", "decoder", "self-attention"],
    },
    {
        "id": "Q02",
        "type": "answerable",
        "paper": "Attention Is All You Need",
        "page": 4,
        "question": "What is the mathematical formula for Scaled Dot-Product Attention in the Transformer?",
        "expected_answer": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
        "key_evidence": "We compute the matrix of outputs as: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
        "key_terms": ["softmax", "sqrt", "d_k"],
    },
    {
        "id": "Q03",
        "type": "answerable",
        "paper": "Attention Is All You Need",
        "page": 6,
        "question": "What is the per-layer computational complexity of a self-attention layer compared to a recurrent layer?",
        "expected_answer": "Self-attention has per-layer complexity O(n^2 * d), whereas recurrent layers have complexity O(n * d^2), where n is sequence length and d is dimension.",
        "key_evidence": "Table 1 shows Self-Attention has per-layer complexity O(n^2 * d) while Recurrent has O(n * d^2).",
        "key_terms": ["n^2", "o(n", "recurrent"],
    },
    {
        "id": "Q04",
        "type": "answerable",
        "paper": "Attention Is All You Need",
        "page": 4,
        "question": "How does Multi-Head Attention combine information from different representation subspaces?",
        "expected_answer": "It linearly projects queries, keys, and values h times, applies attention in parallel, concatenates outputs, and projects again.",
        "key_evidence": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
        "key_terms": ["project", "parallel", "concatenate"],
    },
    # --- Moshi (2410.00037 v1) ---
    {
        "id": "Q05",
        "type": "answerable",
        "paper": "Moshi",
        "page": 1,
        "question": "What is Moshi and what conversational interaction mode does it enable?",
        "expected_answer": "Moshi is a speech-text foundation model enabling real-time full-duplex spoken dialogue where it simultaneously listens and speaks.",
        "key_evidence": "Moshi is a speech-text foundation model which enables real-time spoken dialogue... it always listens and always generates sound.",
        "key_terms": ["speech", "dialogue", "real-time"],
    },
    {
        "id": "Q06",
        "type": "answerable",
        "paper": "Moshi",
        "page": 7,
        "question": "What is the name and parameter count of the base text language model on which Moshi is built?",
        "expected_answer": "Helium, a 7B-parameter text language model built from scratch.",
        "key_evidence": "Moshi is built on top of Helium, a text LLM which we build from scratch... 7B-parameter Helium language model.",
        "key_terms": ["helium", "7b"],
    },
    {
        "id": "Q07",
        "type": "answerable",
        "paper": "Moshi",
        "page": 10,
        "question": "What neural audio codec does Moshi use and what quantization method does it implement?",
        "expected_answer": "Mimi, using split residual vector quantization (RVQ).",
        "key_evidence": "Architecture and training of Mimi, our neural audio codec, with its split residual vector quantization.",
        "key_terms": ["mimi", "residual vector quantization"],
    },
    {
        "id": "Q08",
        "type": "answerable",
        "paper": "Moshi",
        "page": 6,
        "question": "How does Moshi structure user and system audio streams during dialogue?",
        "expected_answer": "It models user and system audio as separate parallel audio codec streams, predicting text and audio tokens jointly.",
        "key_evidence": "Moshi joint audio-text stream architecture using separate audio streams for user and system dialogue.",
        "key_terms": ["stream", "user", "audio"],
    },
    # --- Dynamic Chunking (2507.07955 v2) ---
    {
        "id": "Q09",
        "type": "answerable",
        "paper": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "page": 1,
        "question": "What is dynamic chunking in hierarchical sequence modeling?",
        "expected_answer": "An adaptive segmentation method that learns variable-length chunk boundaries dynamically rather than using fixed heuristic tokenization.",
        "key_evidence": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling... replaces heuristic subword tokenization with learned, adaptive chunk boundaries.",
        "key_terms": ["adaptive", "boundar", "chunk"],
    },
    {
        "id": "Q10",
        "type": "answerable",
        "paper": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "page": 3,
        "question": "What module determines boundary decisions in dynamic chunking?",
        "expected_answer": "The Routing Module, which predicts boundary positions using confidence scores.",
        "key_evidence": "Routing Module... The coefficient c quantifies the routing module's confidence in its boundary decisions.",
        "key_terms": ["routing module", "boundar"],
    },
    {
        "id": "Q11",
        "type": "answerable",
        "paper": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "page": 3,
        "question": "What is the function of the smoothing module in dynamic chunking?",
        "expected_answer": "It smooths boundary decisions to make hierarchical chunking differentiable and stable for end-to-end training.",
        "key_evidence": "Smoothing module... Upper stages w/ Smoothing Module vs w/o Smoothing Module.",
        "key_terms": ["smooth", "boundar", "differentiable"],
    },
    {
        "id": "Q12",
        "type": "answerable",
        "paper": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "page": 6,
        "question": "How does model width scale across hierarchical stages in dynamic chunking?",
        "expected_answer": "Model width scales monotonically across stages: D_0 <= D_1 <= ... <= D_S, allocating more compute and capacity to higher stages.",
        "key_evidence": "First, we ensure the model width is monotone in the hierarchy: D_0 <= D_1 <= ... <= D_S.",
        "key_terms": ["monotone", "d_0", "width"],
    },
    # --- Unanswerable Negative Controls ---
    {
        "id": "Q13",
        "type": "unanswerable",
        "paper": "None",
        "page": "None",
        "question": "What are the recipe ingredients and oven temperature for baking French chocolate croissants?",
        "expected_answer": "The answer cannot be found in the provided research papers.",
        "key_evidence": "Out of domain; not present in papers.",
        "key_terms": ["cannot be found", "not found"],
    },
    {
        "id": "Q14",
        "type": "unanswerable",
        "paper": "None",
        "page": "None",
        "question": "How does the surface code perform fault-tolerant quantum syndrome extraction?",
        "expected_answer": "The answer cannot be found in the provided research papers.",
        "key_evidence": "Out of domain; not present in papers.",
        "key_terms": ["cannot be found", "not found"],
    },
    {
        "id": "Q15",
        "type": "unanswerable",
        "paper": "None",
        "page": "None",
        "question": "What reinforcement learning algorithm and training data were used for OpenAI's GPT-4o voice model?",
        "expected_answer": "The answer cannot be found in the provided research papers.",
        "key_evidence": "Out of domain; not present in papers.",
        "key_terms": ["cannot be found", "not found"],
    },
]


def grade_answer(
    question_type: str,
    actual_answer: str,
    key_terms: List[str],
    is_valid_refusal_bool: bool,
    validator_is_valid: bool,
) -> str:
    """Assign exactly one grade: correct, wrong, refused_correctly, or refused_wrongly."""
    ans_lower = actual_answer.lower()

    if question_type == "unanswerable":
        if is_valid_refusal_bool or any(refusal in ans_lower for refusal in ["cannot be found", "not found", "do not contain"]):
            return "refused_correctly"
        return "wrong"

    # Answerable question
    if is_valid_refusal_bool or any(refusal in ans_lower for refusal in ["cannot be found", "not found", "do not contain"]):
        return "refused_wrongly"

    # Must contain essential factual concept keywords
    matches = sum(1 for term in key_terms if term.lower() in ans_lower)
    term_coverage = matches / len(key_terms) if key_terms else 0.0

    if term_coverage >= 0.60 and validator_is_valid:
        return "correct"
    return "wrong"


def verify_citation(
    actual_answer: str,
    expected_paper: str,
    expected_page: Any,
) -> Tuple[str, Any, bool, bool]:
    """Check whether citation refers to the correct paper and page number."""
    if expected_paper == "None":
        return "", "", True, True

    citations = parse_citations(actual_answer)
    if not citations:
        return "", "", False, False

    top_cit = citations[0]
    cited_paper = top_cit.get("paper_name", "")
    cited_page = top_cit.get("page_number", "N/A")

    paper_match = matches_paper_name(cited_paper, expected_paper)

    page_match = False
    if expected_page != "None" and str(cited_page).isdigit():
        page_match = abs(int(cited_page) - int(expected_page)) <= 1

    return cited_paper, cited_page, paper_match, page_match


async def evaluate_single_item(
    item: Dict[str, Any],
    source: str,
    model,
    index,
    metadata,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run one question through the RAG pipeline and evaluate against ground truth."""
    t_start = time.perf_counter()

    # 1. Retrieval
    t_retrieval = time.perf_counter()
    chunks = search_papers.search(query=item["question"], model=model, index=index, metadata=metadata, top_k=top_k)
    retr_ms = round((time.perf_counter() - t_retrieval) * 1000, 2)

    # 2. Generation
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
    user_prompt = f"Context Passages:\n{context_text}\n\nQuestion: {item['question']}\n\nAnswer with citations:"
    prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

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

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # 3. Groundedness validation
    val = validate_citations(answer_text, chunks)

    # 4. Evidence-based answer grading
    try:
        from rag.citation_validator import is_valid_refusal
    except ImportError:
        from citation_validator import is_valid_refusal
    refusal_flag = is_valid_refusal(answer_text)
    grade = grade_answer(
        question_type=item["type"],
        actual_answer=answer_text,
        key_terms=item["key_terms"],
        is_valid_refusal_bool=refusal_flag,
        validator_is_valid=val["is_valid"],
    )

    # 5. Citation verification
    cited_paper, cited_page, paper_ok, page_ok = verify_citation(
        actual_answer=answer_text,
        expected_paper=item["paper"],
        expected_page=item["page"],
    )

    return {
        "question_id": item["id"],
        "question_type": item["type"],
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "expected_paper": item["paper"],
        "expected_page": item["page"],
        "actual_answer": answer_text,
        "cited_paper": cited_paper,
        "cited_page": cited_page,
        "citation_paper_correct": paper_ok,
        "citation_page_correct": page_ok,
        "answer_grade": grade,
        "retrieval_setting": f"top_k={top_k}",
        "retrieval_latency_ms": retr_ms,
        "total_latency_ms": total_ms,
        "tokens_in": in_tokens,
        "tokens_out": out_tokens,
        "is_valid": val["is_valid"],
        "validation_status": val["validation_status"],
    }


def save_evaluation_csv(results: List[Dict[str, Any]], filepath: str = "answer_evaluation.csv"):
    """Write detailed evaluation results to CSV."""
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDNAMES})
    print(f"\n📁 Saved {len(results)} evaluation records to {filepath}")


def print_evaluation_summary(results: List[Dict[str, Any]]):
    """Print an aggregated statistical summary of answer grades and citation accuracy."""
    total = len(results)
    if total == 0:
        return

    correct_cnt = sum(1 for r in results if r["answer_grade"] == "correct")
    wrong_cnt = sum(1 for r in results if r["answer_grade"] == "wrong")
    refused_correctly_cnt = sum(1 for r in results if r["answer_grade"] == "refused_correctly")
    refused_wrongly_cnt = sum(1 for r in results if r["answer_grade"] == "refused_wrongly")

    overall_accuracy = round(((correct_cnt + refused_correctly_cnt) / total) * 100, 1)

    answerable_rows = [r for r in results if r["question_type"] == "answerable"]
    citations_checked = [r for r in answerable_rows if r["cited_paper"] != ""]
    accurate_citations = sum(1 for r in citations_checked if r["citation_paper_correct"] and r["citation_page_correct"])
    cit_accuracy = round((accurate_citations / len(citations_checked) * 100), 1) if citations_checked else 100.0

    print("\n" + "=" * 66)
    print("               ANSWER QUALITY EVALUATION SUMMARY")
    print("=" * 66)
    print(f"Total Questions Evaluated : {total}")
    print(f"Correct Answers           : {correct_cnt} ({round(correct_cnt/total*100, 1)}%)")
    print(f"Wrong Answers             : {wrong_cnt} ({round(wrong_cnt/total*100, 1)}%)")
    print(f"Correct Refusals          : {refused_correctly_cnt} ({round(refused_correctly_cnt/total*100, 1)}%)")
    print(f"Wrong Refusals            : {refused_wrongly_cnt} ({round(refused_wrongly_cnt/total*100, 1)}%)")
    print("-" * 66)
    print(f"Citation Accuracy         : {cit_accuracy}% ({accurate_citations}/{len(citations_checked)} accurate)")
    print(f"Overall Answer Accuracy   : {overall_accuracy}%")
    print("=" * 66 + "\n")


async def main():
    """Main CLI runner for answer quality evaluation."""
    parser = argparse.ArgumentParser(description="Answer Quality Evaluation Runner")
    parser.add_argument(
        "--source",
        choices=["stand-in", "real"],
        default="stand-in",
        help="Provider: 'stand-in' (default) or 'real'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top_k (default: 5).",
    )
    parser.add_argument(
        "--output",
        default="answer_evaluation.csv",
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    model, index, metadata = answer_generator.load_retrieval_pipeline()
    all_results = []

    print(f"\n🚀 Evaluating {len(EVALUATION_DATASET)} benchmark questions (Source: {args.source}, Top-K: {args.top_k})")

    for item in EVALUATION_DATASET:
        res = await evaluate_single_item(
            item=item,
            source=args.source,
            model=model,
            index=index,
            metadata=metadata,
            top_k=args.top_k,
        )
        all_results.append(res)
        print(f"  [{item['id']}] Grade: {res['answer_grade']:<18} | Paper: {res['expected_paper'][:15]}... | Cit: {res['citation_paper_correct']}")

    save_evaluation_csv(all_results, filepath=args.output)
    print_evaluation_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
