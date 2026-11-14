"""Unit and integration tests for Phase 2: Autonomous Academic Search."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from agent.connectors.base import AcademicPaper
from agent.connectors.arxiv_connector import ArxivConnector
from agent.connectors.crossref_connector import CrossrefConnector
from agent.connectors.semantic_scholar import SemanticScholarConnector
from agent.query_planner import QueryPlanner, generate_fallback_queries
from agent.academic_search import (
    normalize_title,
    calculate_relevance_score,
    deduplicate_and_merge_papers,
    AcademicSearchService,
)

client = TestClient(app)


def test_query_planner_fallback():
    """Verify deterministic query expansion generates valid academic search queries."""
    q = "What is dynamic chunking in sequence modeling and how does it compare to fixed chunking?"
    queries = generate_fallback_queries(q)
    assert len(queries) >= 2
    assert any("dynamic chunking" in query.lower() for query in queries)


def test_query_planner_with_mock():
    """Verify QueryPlanner respects use_mock=True."""
    planner = QueryPlanner()
    queries = planner.plan_queries("How do self-attention mechanisms scale with sequence length?", use_mock=True)
    assert len(queries) >= 1
    assert all(isinstance(x, str) for x in queries)


def test_arxiv_atom_parsing():
    """Verify ArxivConnector parses Atom XML entries into AcademicPaper models."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/1706.03762v7</id>
        <title>Attention Is All You Need</title>
        <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
        <published>2017-06-12T17:57:34Z</published>
        <author><name>Ashish Vaswani</name></author>
        <author><name>Noam Shazeer</name></author>
        <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" type="application/pdf"/>
        <arxiv:doi>10.5555/3295222.3295349</arxiv:doi>
      </entry>
    </feed>
    """
    conn = ArxivConnector()
    papers = conn._parse_atom_feed(sample_xml)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "1706.03762"
    assert p.title == "Attention Is All You Need"
    assert len(p.authors) == 2
    assert p.authors[0] == "Ashish Vaswani"
    assert p.year == 2017
    assert p.pdf_url == "http://arxiv.org/pdf/1706.03762v7"
    assert "recurrent" in p.abstract


def test_crossref_strip_jats_tags():
    """Verify CrossrefConnector strips XML/JATS formatting from abstracts."""
    conn = CrossrefConnector()
    raw = "<jats:title>Abstract</jats:title><jats:p>This paper presents a novel <jats:italic>self-attention</jats:italic> mechanism.</jats:p>"
    clean = conn._strip_jats_tags(raw)
    assert "<" not in clean
    assert ">" not in clean
    assert "This paper presents a novel self-attention mechanism." in clean


def test_title_normalization_and_deduplication():
    """Verify deduplicate_and_merge_papers correctly collapses identical papers from multiple sources."""
    p1 = AcademicPaper(
        paper_id="arxiv_1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        abstract="Short abstract",
        year=2017,
        arxiv_id="1706.03762",
        sources=["arxiv"],
    )
    p2 = AcademicPaper(
        paper_id="crossref_10.5555_3295222",
        title="Attention Is All You Need: Advanced Edition",  # Same base or DOI match
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="A much longer and comprehensive abstract of the Transformer model.",
        year=2017,
        doi="10.5555/3295222",
        arxiv_id="1706.03762",  # Identical arXiv ID triggers merge
        sources=["crossref"],
    )

    merged = deduplicate_and_merge_papers([p1, p2], query_terms=["attention", "transformer"])
    assert len(merged) == 1
    m = merged[0]
    assert set(m.sources) == {"arxiv", "crossref"}
    assert len(m.authors) == 2
    assert "comprehensive abstract" in m.abstract


def test_academic_search_service_mock_mode():
    """Verify AcademicSearchService coordinates discovery without crashing on mock."""
    import asyncio

    async def run_search():
        service = AcademicSearchService()
        res = await service.search(
            question="What is self-attention?",
            sources=["arxiv", "crossref"],
            max_results_per_source=1,
            use_mock=True,
        )
        assert res["status"] == "success"
        assert "arxiv" in res["sources_queried"]
        assert "crossref" in res["sources_queried"]
        assert isinstance(res["papers"], list)
        assert res["total_papers_found"] >= 0

    asyncio.run(run_search())


def test_api_research_sources_endpoint():
    """Verify GET /research/sources returns the supported academic providers."""
    res = client.get("/research/sources")
    assert res.status_code == 200
    data = res.json()
    assert "sources" in data
    source_ids = [s["id"] for s in data["sources"]]
    assert "arxiv" in source_ids
    assert "semantic_scholar" in source_ids
    assert "crossref" in source_ids


def test_api_research_search_endpoint():
    """Verify POST /research/search endpoint validates payloads and executes search."""
    # 1. Invalid empty question
    bad_res = client.post("/research/search", json={"question": "   "})
    assert bad_res.status_code == 400

    # 2. Valid research question with use_mock=True for fast deterministic test
    good_res = client.post(
        "/research/search",
        json={
            "question": "What is dynamic chunking in sequence modeling?",
            "sources": ["arxiv"],
            "max_results_per_source": 2,
            "use_mock": True,
        },
    )
    assert good_res.status_code == 200
    data = good_res.json()
    assert data["status"] == "success"
    assert len(data["generated_queries"]) >= 1
    assert "arxiv" in data["sources_queried"]
    assert isinstance(data["papers"], list)
    if data["papers"]:
        first = data["papers"][0]
        assert "title" in first
        assert "authors" in first
        assert "abstract" in first
