"""Test suite verifying the three retrieval configurations, answer grading, and citation verification."""

import os
import asyncio
import numpy as np
import faiss
from chunk_papers import build_chunks_for_size
from search_papers import search, hybrid_rerank, compute_lexical_overlap
import answer_generator
from evaluate_answers import (
    EVALUATION_DATASET,
    grade_answer,
    verify_citation,
    evaluate_single_item,
    save_evaluation_csv,
    CSV_FIELDNAMES,
)

TEST_EVAL_CSV = "test_answer_evaluation_audit.csv"


def test_chunking_and_retrieval_configurations():
    """Verify that Setting A (250t), Setting B (500t), and Setting C (500t + Hybrid Rerank) work properly."""
    print("=" * 66)
    print("  1. TESTING RETRIEVAL CONFIGURATIONS (SETTING A, B, C)")
    print("=" * 66)

    # 1. Test 250-token chunk generation
    chunks_250 = build_chunks_for_size(chunk_size=250, overlap=25)
    print(f"Setting A Chunks Generated : {len(chunks_250)} chunks")
    assert len(chunks_250) > 215, "250-token chunks should be more numerous than 500-token chunks"
    avg_tokens_250 = sum(c["token_count"] for c in chunks_250) / len(chunks_250)
    print(f"Setting A Avg Token Count  : {avg_tokens_250:.1f} tokens (~250 target)")
    assert 150 < avg_tokens_250 < 300

    # 2. Test Setting B & C using 500-token pipeline
    model, index_500, metadata_500 = answer_generator.load_retrieval_pipeline()
    query = "What is the Transformer architecture?"

    # Setting B: Top-K = 4, No reranking
    res_b = search(query=query, model=model, index=index_500, metadata=metadata_500, top_k=4)
    print(f"\nSetting B (500t, Top-4, No-Rerank): Retrieved {len(res_b)} chunks")
    assert len(res_b) == 4
    assert any(r["paper_name"] == "Attention Is All You Need" for r in res_b)
    print("   Setting B: PASS ✅")

    # Setting C: Top-K = 8 with Hybrid Reranking
    candidates = search(query=query, model=model, index=index_500, metadata=metadata_500, top_k=12)
    res_c = hybrid_rerank(query=query, candidates=candidates, top_k=8)
    print(f"\nSetting C (500t, Top-8, Hybrid-Rerank): Retrieved {len(res_c)} reranked chunks")
    assert len(res_c) == 8
    assert "semantic_score" in res_c[0] and "lexical_score" in res_c[0]
    print(f"   Top-1 Hybrid Score: {res_c[0]['score']} (Sem: {res_c[0]['semantic_score']}, Lex: {res_c[0]['lexical_score']})")
    assert res_c[0]["score"] > 0.0
    print("   Setting C: PASS ✅\n")


def test_answer_grading_logic():
    """Verify that the 4 mutually exclusive answer grades are assigned accurately."""
    print("=" * 66)
    print("  2. TESTING 4-TIER ANSWER GRADING ENGINE")
    print("=" * 66)

    key_terms = ["encoder", "decoder", "self-attention"]

    # Grade 1: correct
    g1 = grade_answer(
        question_type="answerable",
        actual_answer="The Transformer uses an encoder-decoder architecture with stacked self-attention [Attention Is All You Need, p. 3].",
        key_terms=key_terms,
        is_valid_refusal_bool=False,
        validator_is_valid=True,
    )
    print(f"Test correct answer          -> Grade: {g1}")
    assert g1 == "correct"

    # Grade 2: wrong
    g2 = grade_answer(
        question_type="answerable",
        actual_answer="The Transformer was invented in 1895 by space aliens.",
        key_terms=key_terms,
        is_valid_refusal_bool=False,
        validator_is_valid=False,
    )
    print(f"Test wrong answer            -> Grade: {g2}")
    assert g2 == "wrong"

    # Grade 3: refused_correctly
    g3 = grade_answer(
        question_type="unanswerable",
        actual_answer="The answer cannot be found in the provided research papers.",
        key_terms=["cannot be found"],
        is_valid_refusal_bool=True,
        validator_is_valid=True,
    )
    print(f"Test correct refusal         -> Grade: {g3}")
    assert g3 == "refused_correctly"

    # Grade 4: refused_wrongly
    g4 = grade_answer(
        question_type="answerable",
        actual_answer="The answer cannot be found in the provided research papers.",
        key_terms=key_terms,
        is_valid_refusal_bool=True,
        validator_is_valid=True,
    )
    print(f"Test wrongly refused answer  -> Grade: {g4}")
    assert g4 == "refused_wrongly"

    print("   Answer Grading Logic: PASS ✅\n")


def test_citation_verification_logic():
    """Verify that citations are checked against expected paper names and page numbers."""
    print("=" * 66)
    print("  3. TESTING CITATION VERIFICATION")
    print("=" * 66)

    # 1. Matching paper and page
    ans_correct = "The Transformer uses an encoder [Attention Is All You Need, p. 3]."
    p1, pg1, ok_p1, ok_pg1 = verify_citation(ans_correct, "Attention Is All You Need", 3)
    print(f"Accurate Citation   : paper={p1} (ok={ok_p1}), page={pg1} (ok={ok_pg1})")
    assert ok_p1 is True
    assert ok_pg1 is True

    # 2. Mismatched page
    ans_bad_page = "The Transformer uses an encoder [Attention Is All You Need, p. 99]."
    p2, pg2, ok_p2, ok_pg2 = verify_citation(ans_bad_page, "Attention Is All You Need", 3)
    print(f"Bad Page Citation   : paper={p2} (ok={ok_p2}), page={pg2} (ok={ok_pg2})")
    assert ok_p2 is True
    assert ok_pg2 is False

    # 3. Nonexistent paper
    ans_bad_paper = "The model uses an encoder [Unknown Paper 2026, p. 3]."
    p3, pg3, ok_p3, ok_pg3 = verify_citation(ans_bad_paper, "Attention Is All You Need", 3)
    print(f"Bad Paper Citation  : paper={p3} (ok={ok_p3}), page={pg3} (ok={ok_pg3})")
    assert ok_p3 is False

    print("   Citation Verification: PASS ✅\n")


def test_evaluation_pipeline_and_csv():
    """Run sample evaluation items through pipeline and validate CSV schema."""
    print("=" * 66)
    print("  4. TESTING EVALUATION PIPELINE & CSV GENERATION")
    print("=" * 66)

    model, index, metadata = answer_generator.load_retrieval_pipeline()
    sample_items = [EVALUATION_DATASET[0], EVALUATION_DATASET[12]]  # Q01 (answerable) and Q13 (unanswerable)

    results = []
    for item in sample_items:
        res = asyncio.run(evaluate_single_item(item, source="stand-in", model=model, index=index, metadata=metadata))
        results.append(res)
        print(f"Evaluated [{item['id']}]: Grade={res['answer_grade']}, Citations Paper Correct={res['citation_paper_correct']}")

    assert results[0]["answer_grade"] == "correct"
    assert results[1]["answer_grade"] == "refused_correctly"

    save_evaluation_csv(results, filepath=TEST_EVAL_CSV)
    assert os.path.exists(TEST_EVAL_CSV)

    import csv
    with open(TEST_EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDNAMES

    os.remove(TEST_EVAL_CSV)
    print("   Evaluation Pipeline & CSV: PASS ✅\n")


def main():
    test_chunking_and_retrieval_configurations()
    test_answer_grading_logic()
    test_citation_verification_logic()
    test_evaluation_pipeline_and_csv()
    print("=" * 66)
    print("  🎉 ALL RETRIEVAL & ANSWER EVALUATION TESTS PASSED!")
    print("=" * 66)


if __name__ == "__main__":
    main()
