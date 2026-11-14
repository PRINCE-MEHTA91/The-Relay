"""Unit and integration tests for Phase 3: Autonomous Research & Evidence Synthesis."""

import asyncio
import pytest
import fitz
from fastapi.testclient import TestClient

from app.main import app
from agent.connectors.base import AcademicPaper
from agent.document_processor import (
    extract_pdf_pages,
    process_academic_paper,
    chunk_academic_paper,
)
from agent.research_retriever import QueryScopedRetriever
from agent.research_synthesizer import (
    format_evidence_passages,
    validate_research_citations,
    synthesize_research_answer,
    generate_mock_research_answer,
)
from agent.research_coordinator import AutonomousResearchCoordinator

client = TestClient(app)


def create_sample_pdf_bytes() -> bytes:
    """Create a minimal 2-page PDF in-memory for testing."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Page 1: Transformers replace recurrence with multi-head self-attention mechanisms.")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Page 2: Scaled dot-product attention computes softmax(QK^T / sqrt(d_k))V.")
    b = doc.tobytes()
    doc.close()
    return b


def test_pdf_page_extraction_in_memory():
    """Verify PyMuPDF extracts text preserving accurate page numbers."""
    pdf_bytes = create_sample_pdf_bytes()
    pages = extract_pdf_pages(pdf_bytes)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "Transformers replace recurrence" in pages[0][1]
    assert pages[1][0] == 2
    assert "Scaled dot-product attention" in pages[1][1]


def test_process_paper_fallback_to_abstract():
    """Verify paper without a PDF URL cleanly falls back to abstract-only mode."""
    paper = AcademicPaper(
        paper_id="test_001",
        title="Sample Abstract Only Paper",
        authors=["Alice Smith"],
        abstract="This paper introduces a novel adaptive routing mechanism for sequence models.",
        year=2024,
        pdf_url=None,  # No PDF link
    )

    async def run():
        content_type, msg, pages = await process_academic_paper(paper)
        assert content_type == "abstract_only"
        assert "abstract" in msg.lower()
        assert len(pages) == 1
        assert pages[0][0] == 1
        assert "adaptive routing" in pages[0][1]

    asyncio.run(run())


def test_chunk_academic_paper_metadata_preservation():
    """Verify chunk_academic_paper attaches all paper and page metadata to chunks."""
    paper = AcademicPaper(
        paper_id="paper_meta_123",
        title="Hierarchical Sequence Compression",
        authors=["Bob Johnson"],
        year=2025,
        doi="10.1234/hsc.2025",
        arxiv_id="2501.99999",
        url="https://arxiv.org/abs/2501.99999",
        pdf_url="https://arxiv.org/pdf/2501.99999.pdf",
    )
    pages = [
        (1, "Page 1 contains foundational sequence compression principles and hierarchical routing architectures."),
        (2, "Page 2 details dynamic boundary selection modules, differentiable smoothing functions, and empirical latency benchmarks across long sequence lengths."),
    ]
    chunks = chunk_academic_paper(paper, content_type="full_text", pages=pages, chunk_size=15, overlap=5)
    assert len(chunks) >= 2
    c0 = chunks[0]
    assert c0["paper_id"] == "paper_meta_123"
    assert c0["paper_title"] == "Hierarchical Sequence Compression"
    assert c0["authors"] == ["Bob Johnson"]
    assert c0["year"] == 2025
    assert c0["content_type"] == "full_text"
    assert c0["is_abstract_only"] is False
    assert c0["page_number"] in (1, 2)


def test_query_scoped_retriever_cross_query_isolation():
    """Verify that QueryScopedRetriever does not leak chunks between different query sessions."""
    chunks_a = [
        {
            "chunk_id": "a1",
            "paper_title": "Quantum Attention Networks",
            "text": "Quantum circuits simulate entanglement in transformer attention layers.",
            "content_type": "full_text",
            "page_number": 3,
        }
    ]
    chunks_b = [
        {
            "chunk_id": "b1",
            "paper_title": "Baking French Croissants",
            "text": "Lamination of butter and yeast dough creates flaky pastry layers.",
            "content_type": "full_text",
            "page_number": 5,
        }
    ]

    retriever_a = QueryScopedRetriever(chunks_a)
    retriever_b = QueryScopedRetriever(chunks_b)

    results_a = retriever_a.search("quantum entanglement", top_k=5)
    assert len(results_a) == 1
    assert results_a[0]["chunk_id"] == "a1"

    # Croissant search on quantum retriever should return 0 results above threshold
    results_a_croissant = retriever_a.search("croissant butter lamination", top_k=5, score_threshold=0.3)
    assert len(results_a_croissant) == 0

    results_b = retriever_b.search("croissant lamination", top_k=5)
    assert len(results_b) == 1
    assert results_b[0]["chunk_id"] == "b1"


def test_validate_research_citations_full_text_and_abstract():
    """Verify citation validator distinguishes full-text citations from abstract-only citations."""
    papers_used = [
        {"title": "Moshi Speech Model", "content_type": "full_text", "year": 2024, "authors": ["Defossez et al."]},
        {"title": "Dynamic Chunking Overview", "content_type": "abstract_only", "year": 2025, "authors": ["Fang et al."]},
    ]
    evidence_chunks = [
        {"paper_title": "Moshi Speech Model", "page_number": 4, "text": "Moshi uses Mimi audio codec."},
        {"paper_title": "Dynamic Chunking Overview", "page_number": 1, "text": "Dynamic chunking replaces fixed splits."},
    ]

    sample_answer = (
        "Moshi models spoken dialogue with multi-stream codecs [Moshi Speech Model, p. 4]. "
        "In contrast, dynamic segmentation learns boundaries adaptively [Dynamic Chunking Overview (Abstract)]."
    )

    citations, unverified = validate_research_citations(sample_answer, evidence_chunks, papers_used)
    assert len(unverified) == 0
    assert len(citations) == 2

    # Verify citation attributes
    cit_dict = {c["paper_title"]: c for c in citations}
    assert "Moshi Speech Model" in cit_dict
    assert cit_dict["Moshi Speech Model"]["page_number"] == "4"
    assert cit_dict["Moshi Speech Model"]["evidence_type"] == "full_text"

    assert "Dynamic Chunking Overview" in cit_dict
    assert cit_dict["Dynamic Chunking Overview"]["evidence_type"] == "abstract_only"


def test_synthesize_research_answer_insufficient_evidence():
    """Verify synthesizer returns explicit refusal and limitation when no evidence chunks are found."""
    res = synthesize_research_answer(
        question="What is the internal mechanism of non-existent algorithm XYZ?",
        evidence_chunks=[],
        papers_used=[],
        use_mock=True,
    )
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert "does not contain sufficient" in res["answer"]
    assert len(res["limitations"]) > 0


def test_api_research_answer_endpoint():
    """Verify POST /research/answer executes end-to-end autonomous research in mock mode."""
    # 1. Invalid empty question
    bad = client.post("/research/answer", json={"question": "   "})
    assert bad.status_code == 400

    # 2. Valid research question with use_mock=True
    resp = client.post(
        "/research/answer",
        json={
            "question": "How do attention mechanisms compare with dynamic chunking?",
            "max_papers": 2,
            "sources": ["arxiv"],
            "top_k_evidence": 4,
            "use_mock": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "INSUFFICIENT_EVIDENCE")
    assert "question" in data
    assert "answer" in data
    assert "citations" in data
    assert "papers_used" in data
    assert "evidence_snippets" in data
    assert "limitations" in data
    assert data["execution_time_ms"] > 0
    assert data["provider"] in ("mock", "system", "groq")


def test_author_based_citation_validation():
    """Verify validate_research_citations resolves author-name citations (e.g., [Anshul et al., p. 8])."""
    papers_used = [
        {
            "paper_id": "arxiv_2511.10212v1",
            "title": "Next-Frame Feature Prediction for Multimodal Deepfake Detection and Temporal Localization",
            "authors": ["Anshul", "Ritu Garg", "Saurabh Mukherjee"],
            "year": 2025,
            "url": "https://arxiv.org/abs/2511.10212v1",
            "pdf_url": "https://arxiv.org/pdf/2511.10212v1",
            "content_type": "full_text",
        },
        {
            "paper_id": "crossref_1523983",
            "title": "Deepfake Video Detection Using Convolutional Neural Network Based Hybrid Approach",
            "authors": ["Yusuf Koçak", "Ali Kemal Çakır"],
            "year": 2025,
            "url": "https://doi.org/10.2339/politeknik.1523983",
            "pdf_url": None,
            "content_type": "abstract_only",
        },
    ]

    evidence_chunks = [
        {
            "chunk_id": "c1",
            "paper_title": "Next-Frame Feature Prediction for Multimodal Deepfake Detection and Temporal Localization",
            "authors": ["Anshul", "Ritu Garg", "Saurabh Mukherjee"],
            "page_number": 8,
            "content_type": "full_text",
            "text": "A causal transformer decoder predicts cross-modal temporal features.",
        },
        {
            "chunk_id": "c2",
            "paper_title": "Deepfake Video Detection Using Convolutional Neural Network Based Hybrid Approach",
            "authors": ["Yusuf Koçak", "Ali Kemal Çakır"],
            "page_number": 1,
            "content_type": "abstract_only",
            "text": "The hybrid approach achieves high classification accuracy.",
        },
    ]

    answer_text = (
        "Transformer encoders predict cross-modal residuals [Anshul et al., p. 8]. "
        "In contrast, CNN backbones extract frame-level features [Koçak et al., Abstract]."
    )

    citations, unverified = validate_research_citations(answer_text, evidence_chunks, papers_used)
    assert len(unverified) == 0, f"Expected 0 unverified citations, got: {unverified}"
    assert len(citations) == 2
    cit_dict = {c["raw_citation"]: c for c in citations}
    assert "[Anshul et al., p. 8]" in cit_dict
    assert cit_dict["[Anshul et al., p. 8]"]["verified"] is True
    assert cit_dict["[Anshul et al., p. 8]"]["page_number"] == "8"
    assert cit_dict["[Anshul et al., p. 8]"]["evidence_type"] == "full_text"

    assert "[Koçak et al., Abstract]" in cit_dict
    assert cit_dict["[Koçak et al., Abstract]"]["verified"] is True
    assert cit_dict["[Koçak et al., Abstract]"]["evidence_type"] == "abstract_only"


def test_external_research_routing_vs_ask_refusal():
    """Verify out-of-domain questions are refused by /ask (static 3-paper index) but processed by /research/answer."""
    question = "How is Federated Learning used in healthcare to protect patient privacy?"

    # 1. /ask correctly refuses out-of-domain question
    ask_resp = client.post("/ask", json={"question": question, "top_k": 5, "use_mock": True, "stream": False})
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["status"] == "REFUSAL_VERIFIED"
    assert "The available research papers do not contain sufficient information" in ask_data["answer"]

    # 2. /research/answer with mock mode ingests candidate papers and synthesizes grounded answer
    res_resp = client.post(
        "/research/answer",
        json={
            "question": question,
            "max_papers": 2,
            "sources": ["crossref"],
            "top_k_evidence": 4,
            "use_mock": True,
        },
    )
    assert res_resp.status_code == 200
    res_data = res_resp.json()
    assert res_data["status"] in ("success", "INSUFFICIENT_EVIDENCE")
    # If candidate papers were discovered, the answer is synthesized from external papers
    if res_data["status"] == "success":
        assert len(res_data["papers_used"]) > 0
        assert "The available research papers do not contain sufficient information" not in res_data["answer"]


def test_citation_integrity_abstract_only_rejects_page_numbers():
    """Verify that citing page numbers for an abstract-only paper fails validation and is flagged."""
    papers_used = [
        {
            "title": "Quantum Kernels for Drug Discovery",
            "authors": ["Alice Smith"],
            "year": 2024,
            "content_type": "abstract_only",
            "page_count": 1,
        }
    ]
    evidence_chunks = [
        {
            "paper_title": "Quantum Kernels for Drug Discovery",
            "page_number": 1,
            "content_type": "abstract_only",
            "text": "Quantum kernel alignments demonstrate prospective affinity scoring for molecular ligands.",
        }
    ]

    # Invalid citation: cites page 8 from an abstract-only paper
    bad_answer = "Ligand affinity scoring improves via quantum kernels [Quantum Kernels for Drug Discovery, p. 8]."
    cits_bad, unverified_bad = validate_research_citations(bad_answer, evidence_chunks, papers_used)
    assert len(unverified_bad) == 1
    assert "abstract-only" in unverified_bad[0]
    assert cits_bad[0]["verified"] is False

    # Valid citation: cites abstract notation
    good_answer = "Ligand affinity scoring improves via quantum kernels [Quantum Kernels for Drug Discovery (Abstract)]."
    cits_good, unverified_good = validate_research_citations(good_answer, evidence_chunks, papers_used)
    assert len(unverified_good) == 0
    assert cits_good[0]["verified"] is True
    assert cits_good[0]["evidence_type"] == "abstract_only"


def test_citation_integrity_unverified_invented_metrics():
    """Verify that invented accuracy percentages and unsupported metrics are detected and flagged."""
    papers_used = [
        {
            "title": "Empirical Transformer Scaling",
            "authors": ["Elena Vance"],
            "year": 2024,
            "content_type": "full_text",
            "page_count": 10,
        }
    ]
    evidence_chunks = [
        {
            "paper_title": "Empirical Transformer Scaling",
            "page_number": 2,
            "content_type": "full_text",
            "text": "The transformer models scale compute logarithmically with dataset expansion.",
        }
    ]

    # Fabricated metric: "99.8%" does not appear in the evidence chunk
    hallucinated_answer = "The architecture achieves 99.8% precision across benchmarks [Empirical Transformer Scaling, p. 2]."
    cits, unverified = validate_research_citations(hallucinated_answer, evidence_chunks, papers_used)
    assert len(unverified) >= 1
    assert any("99.8%" in u for u in unverified)
    assert cits[0]["verified"] is False


def test_citation_integrity_paper_index_bounds():
    """Verify that referencing nonexistent paper numbers (e.g. Paper 7 when 2 exist) is flagged."""
    papers_used = [
        {"title": "Paper One", "content_type": "full_text", "page_count": 5},
        {"title": "Paper Two", "content_type": "abstract_only", "page_count": 1},
    ]
    evidence_chunks = [
        {"paper_title": "Paper One", "page_number": 1, "content_type": "full_text", "text": "Paper one evidence text."},
    ]

    out_of_bounds_answer = "According to comparative analysis [Paper 7, p. 3], representations converge."
    cits, unverified = validate_research_citations(out_of_bounds_answer, evidence_chunks, papers_used)
    assert len(unverified) >= 1
    assert any("Paper index '7' does not exist" in u for u in unverified)
    assert cits[0]["verified"] is False


def test_format_evidence_passages_removes_ambiguous_chunk_indexing():
    """Verify format_evidence_passages includes paper inventory and removes misleading chunk index tags."""
    papers = [
        {"title": "Quantum ML", "authors": ["Alice"], "year": 2024, "content_type": "full_text", "page_count": 8},
        {"title": "Bioinformatics", "authors": ["Bob"], "year": 2023, "content_type": "abstract_only", "page_count": 1},
    ]
    chunks = [
        {"chunk_id": "c1", "paper_title": "Quantum ML", "authors": ["Alice"], "page_number": 3, "content_type": "full_text", "text": "chunk one"},
        {"chunk_id": "c2", "paper_title": "Bioinformatics", "authors": ["Bob"], "page_number": 1, "content_type": "abstract_only", "text": "chunk two"},
    ]

    xml = format_evidence_passages(chunks, papers)
    assert '<ingested_academic_papers total_count="2">' in xml
    assert '[Paper 1]: "Quantum ML"' in xml
    assert '[Paper 2]: "Bioinformatics"' in xml
    # Ensure there is no confusing index="1" or index="2" on evidence passages that could be mistaken for papers 3-7
    assert '<evidence_passage index=' not in xml
    assert 'source_paper="Paper 1: Quantum ML"' in xml
    assert 'source_paper="Paper 2: Bioinformatics"' in xml


def test_token_metrics_propagated_in_research_response():
    """Verify POST /research/answer returns input_tokens and output_tokens in response."""
    resp = client.post(
        "/research/answer",
        json={
            "question": "What are the scaling dynamics of transformer models?",
            "max_papers": 1,
            "sources": ["arxiv"],
            "top_k_evidence": 2,
            "use_mock": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "input_tokens" in data
    assert "output_tokens" in data
    assert data["input_tokens"] is not None and data["input_tokens"] > 0
    assert data["output_tokens"] is not None and data["output_tokens"] > 0


def test_missing_abstract_treated_as_metadata_only():
    """Verify that paper with empty or stub abstract (e.g. Crossref metadata only) is classified as metadata_only."""
    # 1. Completely empty abstract
    paper_empty = AcademicPaper(
        paper_id="crossref_001",
        title="Quantum Computing in Drug Development",
        authors=["John Doe"],
        abstract="",
        year=2024,
        pdf_url=None,
    )
    # 2. Stub punctuation-only abstract (like '.' returned by some Crossref records)
    paper_stub = AcademicPaper(
        paper_id="crossref_002",
        title="Quantum Drug Discovery Applications",
        authors=["Jane Doe"],
        abstract=".",
        year=2024,
        pdf_url=None,
    )

    async def run():
        c1, m1, _ = await process_academic_paper(paper_empty)
        assert c1 == "metadata_only"
        assert "metadata only" in m1.lower()

        c2, m2, _ = await process_academic_paper(paper_stub)
        assert c2 == "metadata_only"

    asyncio.run(run())


def test_metadata_only_paper_cannot_be_cited_for_empirical_claims():
    """Verify that citing a paper marked metadata_only fails validation and is flagged."""
    papers_used = [
        {
            "title": "Quantum-Enhanced Drug Discovery",
            "authors": ["Sujithkumar"],
            "year": 2024,
            "content_type": "metadata_only",
            "page_count": 1,
        }
    ]
    evidence_chunks = [
        {
            "paper_title": "Quantum-Enhanced Drug Discovery",
            "page_number": 1,
            "content_type": "metadata_only",
            "text": "Title: Quantum-Enhanced Drug Discovery. Authors: Sujithkumar. (No abstract or full text available).",
        }
    ]

    answer = "The hybrid model predicts drug binding with high precision [Quantum-Enhanced Drug Discovery (Abstract)]."
    citations, unverified = validate_research_citations(answer, evidence_chunks, papers_used)
    assert len(unverified) >= 1
    assert any("metadata only" in u.lower() for u in unverified)
    assert citations[0]["verified"] is False


def test_unsupported_claims_detection_gcn_admet_workforce():
    """Verify that claims about GCN, ADMET, or workforce gaps on papers lacking those topics fail validation."""
    papers_used = [
        {
            "title": "Hybrid Quantum-Classical Algorithms for Molecular Simulation",
            "authors": ["Nesx"],
            "year": 2024,
            "content_type": "abstract_only",
            "page_count": 1,
        }
    ]
    evidence_chunks = [
        {
            "paper_title": "Hybrid Quantum-Classical Algorithms for Molecular Simulation",
            "page_number": 1,
            "content_type": "abstract_only",
            "text": (
                "This study explores the implementation of hybrid quantum-classical algorithms in molecular simulation. "
                "The integration of the Variational Quantum Eigensolver (VQE) demonstrates superior accuracy in predicting "
                "binding affinities and reaction energetics compared to traditional methods."
            ),
        }
    ]

    # VQE and binding affinity should pass
    good_answer = "VQE algorithms enhance binding affinity prediction [Hybrid Quantum-Classical Algorithms for Molecular Simulation (Abstract)]."
    c_good, u_good = validate_research_citations(good_answer, evidence_chunks, papers_used)
    assert len(u_good) == 0
    assert c_good[0]["verified"] is True
    assert c_good[0]["evidence_excerpt"] is not None

    # Unsupported claim about GCN and ADMET profiling
    bad_answer = "Graph Convolutional Networks evaluate ADMET profiling [Hybrid Quantum-Classical Algorithms for Molecular Simulation (Abstract)]."
    c_bad, u_bad = validate_research_citations(bad_answer, evidence_chunks, papers_used)
    assert len(u_bad) >= 1
    assert c_bad[0]["verified"] is False


def test_evidence_excerpt_attached_to_citations():
    """Verify that validated citations attach source-specific evidence snippets."""
    papers_used = [
        {"title": "Moshi Speech Model", "content_type": "full_text", "page_count": 8, "authors": ["Defossez"]}
    ]
    evidence_chunks = [
        {"paper_title": "Moshi Speech Model", "page_number": 4, "content_type": "full_text", "text": "Moshi uses Mimi audio codec to model speech tokens."}
    ]

    answer = "Moshi integrates the Mimi audio codec for tokens [Moshi Speech Model, p. 4]."
    cits, unverified = validate_research_citations(answer, evidence_chunks, papers_used)
    assert len(unverified) == 0
    assert len(cits) == 1
    assert cits[0]["verified"] is True
    assert "Mimi audio codec" in cits[0]["evidence_excerpt"]



