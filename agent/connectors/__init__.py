"""Academic source connectors package for The Relay."""

from agent.connectors.base import BaseAcademicConnector, AcademicPaper
from agent.connectors.arxiv_connector import ArxivConnector
from agent.connectors.semantic_scholar import SemanticScholarConnector
from agent.connectors.crossref_connector import CrossrefConnector

__all__ = [
    "BaseAcademicConnector",
    "AcademicPaper",
    "ArxivConnector",
    "SemanticScholarConnector",
    "CrossrefConnector",
]

