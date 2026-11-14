"""Crossref API connector searching published literature across academic publishers."""

import re
import logging
from typing import List, Optional, Dict, Any
import httpx

from agent.connectors.base import BaseAcademicConnector, AcademicPaper

logger = logging.getLogger(__name__)


class CrossrefConnector(BaseAcademicConnector):
    """Connector for searching Crossref using the polite REST API."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, contact_email: str = "researcher@the-relay.local", timeout_seconds: float = 12.0):
        super().__init__(name="crossref", timeout_seconds=timeout_seconds)
        self.contact_email = contact_email

    def _strip_jats_tags(self, text: str) -> str:
        """Strip JATS XML tags (e.g. <jats:p>, <jats:title>) from Crossref abstracts."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", clean).strip()

    def _extract_year(self, item: Dict[str, Any]) -> Optional[int]:
        for date_key in ("published-print", "published-online", "published", "issued"):
            date_data = item.get(date_key)
            if isinstance(date_data, dict):
                parts = date_data.get("date-parts", [])
                if parts and isinstance(parts[0], list) and len(parts[0]) > 0:
                    try:
                        return int(parts[0][0])
                    except (ValueError, TypeError):
                        pass
        return None

    def _extract_authors(self, author_list: List[Dict[str, Any]]) -> List[str]:
        names: List[str] = []
        for a in author_list:
            if not isinstance(a, dict):
                continue
            family = a.get("family", "").strip()
            given = a.get("given", "").strip()
            name = f"{given} {family}".strip() if given and family else (family or given or a.get("name", ""))
            if name:
                names.append(name)
        return names

    def _extract_pdf_url(self, item: Dict[str, Any]) -> Optional[str]:
        links = item.get("link", [])
        if isinstance(links, list):
            for l in links:
                if isinstance(l, dict) and l.get("content-type") == "application/pdf":
                    return l.get("URL")
        return None

    async def search(self, query: str, max_results: int = 5) -> List[AcademicPaper]:
        """Search Crossref and return normalized AcademicPaper items."""
        clean_q = query.strip()
        if not clean_q:
            return []

        params = {
            "query": clean_q,
            "rows": max(1, min(max_results, 25)),
            "select": "DOI,title,abstract,author,published-print,published-online,published,URL,link",
        }
        headers = {
            "User-Agent": f"TheRelay-ResearchAssistant/2.0 (mailto:{self.contact_email})",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = await client.get(self.BASE_URL, params=params, headers=headers)

            if response.status_code != 200:
                logger.warning(f"Crossref API returned HTTP {response.status_code}")
                return []

            data = response.json()
            items = data.get("message", {}).get("items", [])
            papers: List[AcademicPaper] = []

            for item in items:
                doi = item.get("DOI", "").strip()
                title_list = item.get("title", [])
                title = " ".join(title_list).strip() if isinstance(title_list, list) else str(title_list)
                if not title:
                    continue

                abstract_raw = item.get("abstract", "")
                abstract = self._strip_jats_tags(abstract_raw)
                authors = self._extract_authors(item.get("author", []))
                year = self._extract_year(item)
                url = item.get("URL", f"https://doi.org/{doi}" if doi else None)
                pdf_url = self._extract_pdf_url(item)

                paper = AcademicPaper(
                    paper_id=f"crossref_{doi.replace('/', '_')}" if doi else f"crossref_{hash(title)}",
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    doi=doi or None,
                    url=url,
                    pdf_url=pdf_url,
                    sources=["crossref"],
                    metadata={"crossref_type": item.get("type", "journal-article")},
                )
                papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Error querying Crossref for '{clean_q}': {e}")
            return []

