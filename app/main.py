"""Task 11: FastAPI backend exposing The Relay AI Research Assistant via REST endpoints."""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import answer_generator
from mock_llm import MockMode, mock_service

# Initialize FastAPI application with OpenAPI metadata
app = FastAPI(
    title="The Relay — AI Research Assistant API",
    description="REST API for RAG-powered research document question answering with citation validation.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Input payload for research questions."""
    question: str = Field(..., description="The research question to ask.", min_length=1)
    top_k: int = Field(5, ge=1, le=20, description="Number of passages to retrieve.")
    use_mock: bool = Field(False, description="Use the local Mock LLM service instead of Groq.")
    mock_mode: Optional[MockMode] = Field(None, description="Operational mode for mock LLM if use_mock is True.")


class CitationItem(BaseModel):
    """Citation metadata for verified source passages."""
    raw_citation: Optional[str] = None
    paper_name: str
    page_number: Any
    chunk_id: Any
    verified: Optional[bool] = None
    similarity_score: Optional[float] = None
    source_passage: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation audit outcome."""
    is_valid: bool
    citation_errors: List[str]
    unsupported_claims: List[str]
    validated_citations: List[Dict[str, Any]]
    validation_status: str


class AskResponse(BaseModel):
    """Structured response from the research assistant."""
    question: str
    answer: str
    is_valid: bool
    validation: ValidationResult
    citations: List[Dict[str, Any]]
    retrieved_chunks_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    """Welcome endpoint providing project info."""
    return {
        "message": "Welcome to The Relay — AI Research Assistant API",
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Health check endpoint checking retrieval assets."""
    try:
        model, index, metadata = answer_generator.load_retrieval_pipeline()
        return {
            "status": "healthy",
            "indexed_vectors": index.ntotal,
            "metadata_records": len(metadata),
            "embedding_model": answer_generator.EMBEDDING_MODEL_NAME,
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval system unavailable: {str(e)}",
        )


@app.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask(payload: AskRequest):
    """
    Answer a research question using RAG retrieval, LLM synthesis, and citation validation.
    """
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or only whitespace.",
        )

    try:
        if payload.use_mock or payload.mock_mode:
            # Route through Mock LLM service
            start_t = time.perf_counter()
            model, index, metadata = answer_generator.load_retrieval_pipeline()
            import search_papers
            chunks = search_papers.search(
                query=clean_question,
                model=model,
                index=index,
                metadata=metadata,
                top_k=payload.top_k,
            )

            prompt_messages = [
                {"role": "user", "content": f"Context: {answer_generator.format_context_passages(chunks)}\n\nQuestion: {clean_question}"}
            ]
            mock_res = await mock_service.generate_response(
                messages=prompt_messages,
                mode=payload.mock_mode or MockMode.NORMAL,
            )

            if mock_res["status"] != "success":
                raise HTTPException(
                    status_code=mock_res.get("status_code", 500),
                    detail=mock_res["error"],
                )

            from citation_validator import validate_citations
            val = validate_citations(mock_res["content"], chunks)
            total_lat = round((time.perf_counter() - start_t) * 1000, 2)

            return AskResponse(
                question=clean_question,
                answer=mock_res["content"],
                is_valid=val["is_valid"],
                validation=ValidationResult(**val),
                citations=val["validated_citations"],
                retrieved_chunks_count=len(chunks),
                input_tokens=mock_res["input_tokens"],
                output_tokens=mock_res["output_tokens"],
                latency_ms=total_lat,
                status="verified" if val["is_valid"] else "unverified",
            )
        else:
            # Route through standard RAG pipeline
            result = answer_generator.answer_question(
                question=clean_question,
                top_k=payload.top_k,
            )

            if result["status"] == "error":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result["answer"],
                )

            return AskResponse(
                question=clean_question,
                answer=result["answer"],
                is_valid=result["is_valid"],
                validation=ValidationResult(**result["validation"]),
                citations=result["citations"],
                retrieved_chunks_count=len(result["retrieved_chunks"]),
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                latency_ms=result["latency_ms"],
                status=result["status"],
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}",
        )

