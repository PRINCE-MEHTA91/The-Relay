"""Core services, concurrency management, and answer generation for The Relay."""

from .groq_service import (
    GroqService,
    get_groq_client,
    send_chat_completion,
    estimate_tokens,
    get_token_metrics,
)
from .mock_llm import (
    MockLLMService,
    MockMode,
    MockLLMError,
    MockRateLimitError,
    MockServerError,
    MockTimeoutError,
    mock_service,
)
from .rate_limiter import (
    SlidingWindowRateLimiter,
    ConcurrencyQueueManager,
    limiter,
    concurrency_manager,
)
from .answer_generator import (
    load_retrieval_pipeline,
    answer_question,
    format_context_passages,
    extract_citations,
)

__all__ = [
    "GroqService",
    "get_groq_client",
    "send_chat_completion",
    "estimate_tokens",
    "get_token_metrics",
    "MockLLMService",
    "MockMode",
    "MockLLMError",
    "MockRateLimitError",
    "MockServerError",
    "MockTimeoutError",
    "mock_service",
    "SlidingWindowRateLimiter",
    "ConcurrencyQueueManager",
    "limiter",
    "concurrency_manager",
    "load_retrieval_pipeline",
    "answer_question",
    "format_context_passages",
    "extract_citations",
]

