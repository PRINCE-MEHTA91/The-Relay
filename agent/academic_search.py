"""Academic Search Service orchestrating multi-source discovery, deduplication, and ranking."""

import re
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

from agent.connectors.base import AcademicPaper, BaseAcademicConnector
from agent.connectors.arxiv_connector import ArxivConnector
from agent.connectors.semantic_scholar import SemanticScholarConnector
from agent.connectors.crossref_connector import CrossrefConnector
from agent.query_planner import QueryPlanner

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    """Normalize paper title for deduplication comparison."""
    clean = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", clean).strip()


def calculate_relevance_score(paper: AcademicPaper, query_terms: List[str]) -> float:
    """Calculate a heuristic relevance score based on keyword presence in title and abstract."""
    score = 0.0
    title_lower = paper.title.lower()
    abstract_lower = paper.abstract.lower()

    for term in query_terms:
        term_clean = term.lower().strip()
        if not term_clean:
            continue
        # Title matches carry higher weight
        if term_clean in title_lower:
            score += 2.0
        # Abstract matches
        if term_clean in abstract_lower:
            score += 0.8

    # Substantive abstract bonus (prioritize papers with actual empirical text over metadata stubs)
    if len(paper.abstract.strip()) >= 50:
        score += 2.0
    elif not paper.pdf_url:
        score = max(0.0, score - 3.0)

    # Bonus for open-access PDF availability
    if paper.pdf_url:
        score += 1.5

    # Bonus for presence in multiple academic databases
    if len(paper.sources) > 1:
        score += 1.0

    return round(score, 3)


def deduplicate_and_merge_papers(papers: List[AcademicPaper], query_terms: List[str]) -> List[AcademicPaper]:
    """Merge duplicate papers discovered across different academic sources."""
    merged_papers: Dict[str, AcademicPaper] = {}

    for paper in papers:
        # Check matching keys
        match_key = None

        # 1. Match by DOI
        if paper.doi:
            clean_doi = paper.doi.lower().strip()
            for k, existing in merged_papers.items():
                if existing.doi and existing.doi.lower().strip() == clean_doi:
                    match_key = k
                    break

        # 2. Match by arXiv ID
        if not match_key and paper.arxiv_id:
            clean_arx = paper.arxiv_id.lower().strip()
            for k, existing in merged_papers.items():
                if existing.arxiv_id and existing.arxiv_id.lower().strip() == clean_arx:
                    match_key = k
                    break

        # 3. Match by normalized title
        if not match_key:
            norm_title = normalize_title(paper.title)
            if len(norm_title) >= 15:
                for k, existing in merged_papers.items():
                    if normalize_title(existing.title) == norm_title:
                        match_key = k
                        break

        if match_key:
            # Merge with existing paper
            existing = merged_papers[match_key]
            # Merge sources
            for s in paper.sources:
                if s not in existing.sources:
                    existing.sources.append(s)
            # Take longer abstract if existing is brief
            if len(paper.abstract) > len(existing.abstract):
                existing.abstract = paper.abstract
            # Preserve PDF URL if existing lacks one
            if not existing.pdf_url and paper.pdf_url:
                existing.pdf_url = paper.pdf_url
            # Preserve DOI / arXiv ID
            if not existing.doi and paper.doi:
                existing.doi = paper.doi
            if not existing.arxiv_id and paper.arxiv_id:
                existing.arxiv_id = paper.arxiv_id
            # Merge authors if existing has none or fewer
            if len(paper.authors) > len(existing.authors):
                existing.authors = paper.authors
        else:
            # Register new paper entry
            key = paper.doi or paper.arxiv_id or normalize_title(paper.title) or paper.paper_id
            merged_papers[key] = paper

    results = list(merged_papers.values())
    # Calculate scores and sort by relevance descending
    for p in results:
        p.score = calculate_relevance_score(p, query_terms)

    results.sort(key=lambda x: (x.score or 0.0), reverse=True)
    return results


class AcademicSearchService:
    """Orchestrates multi-source academic paper discovery, query expansion, and deduplication."""

    def __init__(
        self,
        query_planner: Optional[QueryPlanner] = None,
        connectors: Optional[List[BaseAcademicConnector]] = None,
    ):
        self.query_planner = query_planner or QueryPlanner()
        if connectors is not None:
            self.connectors = {c.name: c for c in connectors}
        else:
            self.connectors = {
                "arxiv": ArxivConnector(),
                "semantic_scholar": SemanticScholarConnector(),
                "crossref": CrossrefConnector(),
            }

    async def search(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 5,
        use_mock: bool = False,
    ) -> Dict[str, Any]:
        """Perform autonomous academic search for a research question."""
        t_start = time.perf_counter()
        clean_question = question.strip()
        if not clean_question:
            return {
                "question": "",
                "generated_queries": [],
                "sources_queried": [],
                "papers": [],
                "total_papers_found": 0,
                "execution_time_ms": 0.0,
                "status": "error",
                "message": "Question cannot be empty.",
            }

        # 1. Generate targeted search queries from research question
        generated_queries = self.query_planner.plan_queries(
            question=clean_question,
            use_mock=use_mock,
            max_queries=3,
        )
        if not generated_queries:
            generated_queries = [clean_question]

        # 2. Determine active connectors
        active_sources = sources or ["arxiv", "semantic_scholar", "crossref"]
        target_connectors = [
            self.connectors[s] for s in active_sources if s in self.connectors
        ]
        if not target_connectors:
            target_connectors = list(self.connectors.values())

        # 3. Query all connectors concurrently across all generated queries
        search_tasks = []
        for q in generated_queries:
            for connector in target_connectors:
                search_tasks.append(connector.search(q, max_results=max_results_per_source))

        raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        collected_papers: List[AcademicPaper] = []
        for res in raw_results:
            if isinstance(res, list):
                collected_papers.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Error in academic search task: {res}")

        # 4. Deduplicate, merge, and rank papers
        all_query_terms = []
        for q in generated_queries:
            all_query_terms.extend(q.split())
        all_query_terms.extend(clean_question.split())

        deduped_papers = deduplicate_and_merge_papers(collected_papers, all_query_terms)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "question": clean_question,
            "generated_queries": generated_queries,
            "sources_queried": [c.name for c in target_connectors],
            "papers": [p.model_dump() for p in deduped_papers],
            "total_papers_found": len(deduped_papers),
            "execution_time_ms": duration_ms,
            "status": "success",
        }

