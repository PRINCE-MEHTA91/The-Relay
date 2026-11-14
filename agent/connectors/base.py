"""Base models and abstract interface for academic source connectors."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AcademicPaper(BaseModel):
    """Normalized schema for academic papers across all literature providers."""
    paper_id: str = Field(..., description="Unique paper identifier (arXiv ID, DOI, or hash)")
    title: str = Field(..., description="Full title of the research paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(default="", description="Paper abstract text")
    year: Optional[int] = Field(None, description="Publication year")
    doi: Optional[str] = Field(None, description="Digital Object Identifier (DOI)")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier e.g. 2301.00704")
    url: Optional[str] = Field(None, description="URL to paper landing page")
    pdf_url: Optional[str] = Field(None, description="Direct URL to open-access PDF if available")
    sources: List[str] = Field(default_factory=list, description="Academic sources where discovered")
    score: Optional[float] = Field(None, description="Relevance score for ranking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw source-specific metadata")


class BaseAcademicConnector(ABC):
    """Abstract base class for academic literature search connectors."""

    def __init__(self, name: str, timeout_seconds: float = 12.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[AcademicPaper]:
        """Search the academic provider and return normalized AcademicPaper items."""
        pass

