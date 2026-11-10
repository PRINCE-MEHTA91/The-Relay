""" FastAPI backend with bounded request queue, SSE streaming, and rate limiting."""

import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import answer_generator
from mock_llm import MockMode, mock_service
from rate_limiter import limiter, concurrency_manager

# Initialize FastAPI application with OpenAPI metadata
app = FastAPI(
    title="The Relay — AI Research Assistant API",
    description="REST API for RAG research Q&A with bounded queue, SSE streaming, and citation validation.",
    version="1.0.0",
)


# Pydantic Schemas


class AskRequest(BaseModel):
    """Input payload for research questions."""
    question: str = Field(..., description="The research question to ask.", min_length=1)
    top_k: int = Field(5, ge=1, le=20, description="Number of passages to retrieve.")
    use_mock: bool = Field(False, description="Use the local Mock LLM service instead of Groq.")
    mock_mode: Optional[MockMode] = Field(None, description="Operational mode for mock LLM if use_mock is True.")
    stream: bool = Field(False, description="Stream response progressively via Server-Sent Events (SSE).")


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
    retries_attempted: int = 0
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


@app.get("/metrics", status_code=status.HTTP_200_OK)
def metrics():
    """Return runtime concurrency, queue depth, and rate limiting statistics."""
    stats = concurrency_manager.stats()
    return {
        "rate_limiter": {
            "max_requests": limiter.max_requests,
            "window_seconds": limiter.window_seconds,
        },
        "concurrency": stats,
    }


@app.post("/ask", status_code=status.HTTP_200_OK)
async def ask(payload: AskRequest, request: Request, response: Response):
    """Answer research questions with bounded queueing, rate limiting, and optional SSE streaming."""
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or only whitespace.",
        )

    # 1. Enforce sliding-window rate limit per client IP
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "127.0.0.1")
    allowed, remaining, retry_after = await limiter.check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={
                "Retry-After": str(int(retry_after) + 1),
                "X-RateLimit-Limit": str(limiter.max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            },
        )

    # 2. Enforce bounded concurrency queue
    acquired, reason = await concurrency_manager.acquire()
    if not acquired:
        if reason == "QUEUE_FULL":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server busy: request queue is full. Please try again later.",
                headers={"Retry-After": "5"},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server busy: request queue timed out waiting for an execution slot.",
                headers={"Retry-After": "5"},
            )

    # 3. Handle Streaming Response via Server-Sent Events (SSE)
    if payload.stream:
        try:
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

            async def sse_event_generator():
                try:
                    init_data = json.dumps({"type": "init", "retrieved_chunks": len(chunks), "done": False})
                    yield f"data: {init_data}\n\n"

                    async for token in mock_service.stream_response(prompt_messages, chunk_delay=0.03):
                        if await request.is_disconnected():
                            break
                        chunk_data = json.dumps({"token": token, "done": False})
                        yield f"data: {chunk_data}\n\n"

                        if payload.mock_mode == MockMode.SLOW:
                            await asyncio.sleep(0.1)

                    if not await request.is_disconnected():
                        full_content = mock_service._generate_mock_content(prompt_messages)
                        from citation_validator import validate_citations
                        val = validate_citations(full_content, chunks)
                        in_toks = sum(len(m.get("content", "").split()) for m in prompt_messages)
                        out_toks = len(full_content.split())
                        end_data = json.dumps({
                            "token": "",
                            "done": True,
                            "citations": val["validated_citations"],
                            "status": "verified" if val["is_valid"] else "unverified",
                            "input_tokens": in_toks,
                            "output_tokens": out_toks,
                            "retries_attempted": 0,
                        })
                        yield f"data: {end_data}\n\n"
                finally:
                    concurrency_manager.release()

            return StreamingResponse(
                sse_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-RateLimit-Limit": str(limiter.max_requests),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )
        except Exception:
            concurrency_manager.release()
            raise

    # 4. Handle Standard Normal Response (JSON)
    try:
        response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        if payload.use_mock or payload.mock_mode:
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
                retries_attempted=mock_res.get("retries_attempted", 0),
                latency_ms=total_lat,
                status="verified" if val["is_valid"] else "unverified",
            )
        else:
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
                retries_attempted=result.get("retries_attempted", 0),
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
    finally:
        concurrency_manager.release()


@app.post("/ask/stream", status_code=status.HTTP_200_OK)
async def ask_stream(payload: AskRequest, request: Request, response: Response):
    """Dedicated endpoint returning SSE streaming response."""
    payload.stream = True
    return await ask(payload, request, response)
