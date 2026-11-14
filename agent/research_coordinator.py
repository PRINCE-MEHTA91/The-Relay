"""Coordinator tying together discovery, download, extraction, scoped retrieval, and synthesis."""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

from agent.connectors.base import AcademicPaper
from agent.academic_search import AcademicSearchService
from agent.document_processor import process_academic_paper, chunk_academic_paper
from agent.research_retriever import QueryScopedRetriever
from agent.research_synthesizer import synthesize_research_answer

logger = logging.getLogger(__name__)


class AutonomousResearchCoordinator:
    """Orchestrates end-to-end autonomous research without requiring user PDF uploads."""

    def __init__(self, search_service: Optional[AcademicSearchService] = None):
        self.search_service = search_service or AcademicSearchService()

    async def conduct_research(
        self,
        question: str,
        max_papers: int = 3,
        sources: Optional[List[str]] = None,
        top_k_evidence: int = 8,
        use_mock: bool = False,
    ) -> Dict[str, Any]:
        """Execute autonomous research from user question to grounded, cited answer."""
        t_start = time.perf_counter()
        clean_question = question.strip()

        if not clean_question:
            return {
                "question": "",
                "answer": "Error: Research question cannot be empty.",
                "citations": [],
                "papers_used": [],
                "evidence_snippets": [],
                "limitations": ["Invalid question payload."],
                "execution_time_ms": 0.0,
                "status": "error",
                "provider": "system",
                "model": "input-validator",
            }

        # 1. Academic Paper Discovery across arXiv, Semantic Scholar, and Crossref
        search_res = await self.search_service.search(
            question=clean_question,
            sources=sources,
            max_results_per_source=max(2, max_papers),
            use_mock=use_mock,
        )

        raw_papers = search_res.get("papers", [])
        if not raw_papers:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            insufficient = (
                "No matching academic literature could be discovered across arXiv, Semantic Scholar, or Crossref "
                "for this research question."
            )
            return {
                "question": clean_question,
                "answer": insufficient,
                "citations": [],
                "papers_used": [],
                "evidence_snippets": [],
                "limitations": ["Academic search across arXiv, Semantic Scholar, and Crossref returned 0 candidate papers."],
                "execution_time_ms": duration_ms,
                "status": "INSUFFICIENT_EVIDENCE",
                "provider": "system",
                "model": "literature-discovery",
            }

        # Select top N papers
        candidate_papers = [AcademicPaper(**p) for p in raw_papers[:max_papers]]

        # 2. Document Ingestion (Download PDF or fallback to Abstract)
        ingest_tasks = [
            process_academic_paper(p, timeout_seconds=12.0)
            for p in candidate_papers
        ]
        processed_results = await asyncio.gather(*ingest_tasks, return_exceptions=True)

        all_chunks: List[Dict[str, Any]] = []
        papers_used_metadata: List[Dict[str, Any]] = []

        for paper, res in zip(candidate_papers, processed_results):
            if isinstance(res, Exception):
                logger.error(f"Error processing paper '{paper.title}': {res}")
                content_type = "abstract_only"
                msg = f"Download/parse exception ({str(res)}); defaulted to abstract."
                pages = [(1, paper.abstract or paper.title)]
            else:
                content_type, msg, pages = res

            paper_meta = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "sources": paper.sources,
                "content_type": content_type,
                "ingest_message": msg,
                "page_count": len(pages),
            }
            papers_used_metadata.append(paper_meta)

            # Chunk paper
            chunks = chunk_academic_paper(
                paper=paper,
                content_type=content_type,
                pages=pages,
                chunk_size=400,
                overlap=40,
            )
            all_chunks.extend(chunks)

        # 3. Query-Scoped Vector Retrieval (isolated in-memory FAISS index)
        retriever = QueryScopedRetriever(all_chunks)
        evidence_chunks = retriever.search(
            query=clean_question,
            top_k=top_k_evidence,
            score_threshold=0.15,
        )

        # 4. Evidence-Grounded Research Answer Synthesis & Citation Validation
        synth_result = synthesize_research_answer(
            question=clean_question,
            evidence_chunks=evidence_chunks,
            papers_used=papers_used_metadata,
            use_mock=use_mock,
        )

        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Truncate evidence text for client response snippet preview
        snippets = []
        for c in evidence_chunks:
            snippets.append({
                "chunk_id": c["chunk_id"],
                "paper_title": c["paper_title"],
                "page_number": c["page_number"],
                "content_type": c["content_type"],
                "similarity_score": c["score"],
                "snippet": c["text"][:300].strip() + ("..." if len(c["text"]) > 300 else ""),
            })

        return {
            "question": clean_question,
            "answer": synth_result["answer"],
            "citations": synth_result["citations"],
            "papers_used": papers_used_metadata,
            "evidence_snippets": snippets,
            "limitations": synth_result["limitations"],
            "execution_time_ms": duration_ms,
            "status": synth_result["status"],
            "provider": synth_result["provider"],
            "model": synth_result["model"],
            "input_tokens": synth_result.get("input_tokens"),
            "output_tokens": synth_result.get("output_tokens"),
            "search_queries": search_res.get("generated_queries", []),
            "sources_queried": search_res.get("sources_queried", []),
            "total_papers_found": search_res.get("total_papers_found", len(raw_papers)),
            "total_chunks_created": len(all_chunks),
        }
