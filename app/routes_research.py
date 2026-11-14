"""FastAPI router for Autonomous Research endpoints under /research/*."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from agent.academic_search import AcademicSearchService
from agent.query_planner import QueryPlanner
from agent.research_coordinator import AutonomousResearchCoordinator

router = APIRouter(prefix="/research", tags=["Autonomous Research"])

# Singleton service instances
search_service = AcademicSearchService(query_planner=QueryPlanner())
research_coordinator = AutonomousResearchCoordinator(search_service=search_service)


class AcademicSearchRequest(BaseModel):
    """Payload for autonomous academic search."""
    question: str = Field(..., description="The research question to explore.", min_length=1)
    sources: Optional[List[str]] = Field(
        default=["arxiv", "semantic_scholar", "crossref"],
        description="List of academic repositories to search.",
    )
    max_results_per_source: int = Field(default=5, ge=1, le=20, description="Max results per repository.")
    use_mock: bool = Field(default=False, description="Use deterministic query planning fallback.")


class AcademicSearchResponse(BaseModel):
    """Response containing discovered academic papers and generated queries."""
    question: str
    generated_queries: List[str]
    sources_queried: List[str]
    papers: List[Dict[str, Any]]
    total_papers_found: int
    execution_time_ms: float
    status: str


class ResearchAnswerRequest(BaseModel):
    """Payload for full autonomous research and evidence-based answer generation."""
    question: str = Field(..., description="The natural-language research question.", min_length=1)
    max_papers: int = Field(default=3, ge=1, le=10, description="Number of top papers to ingest and inspect.")
    sources: Optional[List[str]] = Field(
        default=["arxiv", "semantic_scholar", "crossref"],
        description="List of academic sources to query.",
    )
    top_k_evidence: int = Field(default=8, ge=2, le=20, description="Number of evidence chunks to retrieve.")
    use_mock: bool = Field(default=False, description="Use stand-in mock generation instead of Groq LLM.")


class ResearchAnswerResponse(BaseModel):
    """Complete answer synthesis with paper metadata, citations, and research limitations."""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    papers_used: List[Dict[str, Any]]
    evidence_snippets: List[Dict[str, Any]]
    limitations: List[str]
    execution_time_ms: float
    status: str
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    search_queries: Optional[List[str]] = None
    sources_queried: Optional[List[str]] = None
    total_papers_found: Optional[int] = None
    total_chunks_created: Optional[int] = None


@router.post("/search", response_model=AcademicSearchResponse, status_code=status.HTTP_200_OK)
async def search_academic_literature(payload: AcademicSearchRequest):
    """Autonomously search academic literature (arXiv, Semantic Scholar, Crossref) for a research question."""
    clean_q = payload.question.strip()
    if not clean_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or only whitespace.",
        )

    try:
        result = await search_service.search(
            question=clean_q,
            sources=payload.sources,
            max_results_per_source=payload.max_results_per_source,
            use_mock=payload.use_mock,
        )
        return AcademicSearchResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Academic search failed: {str(e)}",
        )


@router.post("/answer", response_model=ResearchAnswerResponse, status_code=status.HTTP_200_OK)
async def answer_research_question(payload: ResearchAnswerRequest):
    """Autonomously discover papers, extract full-text or abstract, retrieve evidence, and synthesize an answer."""
    clean_q = payload.question.strip()
    if not clean_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or only whitespace.",
        )

    try:
        result = await research_coordinator.conduct_research(
            question=clean_q,
            max_papers=payload.max_papers,
            sources=payload.sources,
            top_k_evidence=payload.top_k_evidence,
            use_mock=payload.use_mock,
        )
        return ResearchAnswerResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Autonomous research synthesis failed: {str(e)}",
        )


@router.get("/sources", status_code=status.HTTP_200_OK)
def get_supported_sources():
    """Return available academic repositories and their operational status."""
    return {
        "sources": [
            {
                "id": "arxiv",
                "name": "arXiv",
                "description": "Open access e-prints in Computer Science, AI, Mathematics, and Physics.",
                "enabled": True,
                "coverage": "Preprints, direct open-access PDFs",
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "description": "AI-powered academic search engine by the Allen Institute for AI (AI2).",
                "enabled": True,
                "coverage": "Citation graphs, abstracts, open-access PDF links",
            },
            {
                "id": "crossref",
                "name": "Crossref",
                "description": "Official DOI registry for peer-reviewed journal articles and conference proceedings.",
                "enabled": True,
                "coverage": "Published journals, DOIs, publisher bibliographic metadata",
            },
        ]
    }
