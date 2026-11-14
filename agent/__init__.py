"""Autonomous Research Agent package for The Relay."""

from agent.connectors.base import AcademicPaper, BaseAcademicConnector
from agent.query_planner import QueryPlanner
from agent.academic_search import AcademicSearchService
from agent.document_processor import process_academic_paper, chunk_academic_paper, download_paper_pdf, extract_pdf_pages
from agent.research_retriever import QueryScopedRetriever
from agent.research_synthesizer import synthesize_research_answer, validate_research_citations
from agent.research_coordinator import AutonomousResearchCoordinator

__all__ = [
    "AcademicPaper",
    "BaseAcademicConnector",
    "QueryPlanner",
    "AcademicSearchService",
    "process_academic_paper",
    "chunk_academic_paper",
    "download_paper_pdf",
    "extract_pdf_pages",
    "QueryScopedRetriever",
    "synthesize_research_answer",
    "validate_research_citations",
    "AutonomousResearchCoordinator",
]
