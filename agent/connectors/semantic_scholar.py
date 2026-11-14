"""Semantic Scholar Academic Graph API connector with rate-limit resilience."""

import os
import logging
from typing import List, Optional, Dict, Any
import httpx

from agent.connectors.base import BaseAcademicConnector, AcademicPaper

logger = logging.getLogger(__name__)


class SemanticScholarConnector(BaseAcademicConnector):
    """Connector for searching Semantic Scholar using the Academic Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 10.0):
        super().__init__(name="semantic_scholar", timeout_seconds=timeout_seconds)
        self.api_key = api_key or os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def _extract_authors(self, author_list: Optional[List[Dict[str, Any]]]) -> List[str]:
        if not author_list:
            return []
        names = []
        for a in author_list:
            if isinstance(a, dict) and a.get("name"):
                names.append(a["name"].strip())
        return names

    async def search(self, query: str, max_results: int = 5) -> List[AcademicPaper]:
        """Search Semantic Scholar and return normalized AcademicPaper items."""
        clean_q = query.strip()
        if not clean_q:
            return []

        params = {
            "query": clean_q,
            "limit": max(1, min(max_results, 25)),
            "fields": "paperId,title,abstract,authors,year,externalIds,url,openAccessPdf",
        }
        headers = {
            "User-Agent": "TheRelay-ResearchAssistant/2.0 (mailto:researcher@the-relay.local)",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = await client.get(self.BASE_URL, params=params, headers=headers)

            if response.status_code == 429:
                logger.info("Semantic Scholar unauthenticated quota limit reached (HTTP 429); continuing with other providers.")
                return []
            elif response.status_code != 200:
                logger.warning(f"Semantic Scholar API returned HTTP {response.status_code}")
                return []

            data = response.json()
            items = data.get("data", [])
            papers: List[AcademicPaper] = []

            for item in items:
                paper_id = item.get("paperId", "")
                title = (item.get("title") or "").strip()
                if not title:
                    continue

                abstract = (item.get("abstract") or "").strip()
                authors = self._extract_authors(item.get("authors"))
                year = item.get("year")
                ext_ids = item.get("externalIds") or {}
                doi = ext_ids.get("DOI")
                arxiv_id = ext_ids.get("ArXiv")
                url = item.get("url")

                # Open Access PDF link
                pdf_url = None
                oa_info = item.get("openAccessPdf")
                if isinstance(oa_info, dict) and oa_info.get("url"):
                    pdf_url = oa_info.get("url")
                elif arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                paper = AcademicPaper(
                    paper_id=f"s2_{paper_id}" if paper_id else f"s2_{hash(title)}",
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    url=url,
                    pdf_url=pdf_url,
                    sources=["semantic_scholar"],
                    metadata={"s2_paper_id": paper_id},
                )
                papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Error querying Semantic Scholar for '{clean_q}': {e}")
            return []

