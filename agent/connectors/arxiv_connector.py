"""arXiv API connector searching preprints via Atom XML feed."""

import re
import asyncio
import logging
from typing import List, Optional
import xml.etree.ElementTree as ET
import httpx

from agent.connectors.base import BaseAcademicConnector, AcademicPaper

logger = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivConnector(BaseAcademicConnector):
    """Connector for searching arXiv preprints using the official arXiv API."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout_seconds: float = 15.0):
        super().__init__(name="arxiv", timeout_seconds=timeout_seconds)
        self._lock = asyncio.Lock()
        self._last_call_time = 0.0

    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _extract_arxiv_id(self, raw_id: str) -> str:
        # e.g. http://arxiv.org/abs/2301.00704v1 -> 2301.00704
        match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", raw_id)
        if match:
            return match.group(1)
        # Old arXiv format e.g. hep-th/9901001
        match_old = re.search(r"([a-z\-]+/\d{7})", raw_id)
        if match_old:
            return match_old.group(1)
        return raw_id.split("/")[-1]

    async def search(self, query: str, max_results: int = 5) -> List[AcademicPaper]:
        """Search arXiv and parse results into AcademicPaper objects."""
        clean_q = self._clean_text(query)
        if not clean_q:
            return []

        # Polite rate limiting: brief lock to space arXiv calls
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_call_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            self._last_call_time = asyncio.get_event_loop().time()

        params = {
            "search_query": f"all:{clean_q}",
            "start": 0,
            "max_results": max(1, min(max_results, 25)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        headers = {
            "User-Agent": "TheRelay-ResearchAssistant/2.0 (mailto:research@the-relay.local)",
            "Accept": "application/atom+xml, application/xml",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = await client.get(self.BASE_URL, params=params, headers=headers)

            if response.status_code != 200:
                logger.warning(f"arXiv API returned HTTP {response.status_code}")
                return []

            return self._parse_atom_feed(response.text)
        except Exception as e:
            logger.error(f"Error querying arXiv for '{clean_q}': {e}")
            return []

    def _parse_atom_feed(self, xml_content: str) -> List[AcademicPaper]:
        papers: List[AcademicPaper] = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return []

        entries = root.findall("atom:entry", ATOM_NS)
        for entry in entries:
            # 1. ID
            id_elem = entry.find("atom:id", ATOM_NS)
            raw_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
            arxiv_id = self._extract_arxiv_id(raw_url)

            # 2. Title
            title_elem = entry.find("atom:title", ATOM_NS)
            title = self._clean_text(title_elem.text) if title_elem is not None else "Untitled arXiv Paper"

            # 3. Summary / Abstract
            summary_elem = entry.find("atom:summary", ATOM_NS)
            abstract = self._clean_text(summary_elem.text) if summary_elem is not None else ""

            # 4. Authors
            authors: List[str] = []
            for author_elem in entry.findall("atom:author", ATOM_NS):
                name_elem = author_elem.find("atom:name", ATOM_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(self._clean_text(name_elem.text))

            # 5. Published Year
            published_elem = entry.find("atom:published", ATOM_NS)
            year: Optional[int] = None
            if published_elem is not None and published_elem.text:
                year_match = re.search(r"(\d{4})", published_elem.text)
                if year_match:
                    year = int(year_match.group(1))

            # 6. Links (Landing URL and PDF URL)
            url = raw_url
            pdf_url: Optional[str] = None
            for link_elem in entry.findall("atom:link", ATOM_NS):
                rel = link_elem.get("rel")
                href = link_elem.get("href", "")
                link_title = link_elem.get("title", "")
                if link_title == "pdf" or "pdf" in href:
                    pdf_url = href
                elif rel == "alternate" and href:
                    url = href

            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # 7. DOI if present
            doi_elem = entry.find("arxiv:doi", ATOM_NS)
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

            paper = AcademicPaper(
                paper_id=f"arxiv_{arxiv_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                url=url,
                pdf_url=pdf_url,
                sources=["arxiv"],
                metadata={"raw_entry_id": raw_url},
            )
            papers.append(paper)

        return papers

