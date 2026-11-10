"""Task 8: Groq API client with token budget enforcement, configurable timeout, and exponential retries."""

import os
import time
import random
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import groq
from groq import Groq

load_dotenv()

# Configurable token limits and retry settings
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
DEFAULT_MAX_INPUT_TOKENS = int(os.getenv("GROQ_MAX_INPUT_TOKENS", "4096"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", os.getenv("GROQ_MAX_TOKENS", "1024")))
DEFAULT_TIMEOUT = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30.0"))
DEFAULT_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
DEFAULT_INITIAL_RETRY_DELAY = float(os.getenv("GROQ_INITIAL_RETRY_DELAY", "1.0"))

# In-memory cumulative token tracker for auditability
CUMULATIVE_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "successful_requests": 0,
    "failed_requests": 0,
}


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using standard word-ratio heuristic (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


def get_groq_client() -> Groq:
    """Initialize Groq client using API key from .env; raises ValueError if missing."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or is still the placeholder. "
            "Please add your valid Groq API key to .env file."
        )
    return Groq(api_key=api_key)


def get_available_models(client: Optional[Groq] = None) -> List[str]:
    """Fetch active model IDs from Groq API."""
    if client is None:
        try:
            client = get_groq_client()
        except ValueError:
            return []
    model_list = client.models.list()
    return [m.id for m in model_list.data if getattr(m, "active", True)]


def send_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Send prompt to Groq with token limits, timeout, and retry handling."""
    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    temp = DEFAULT_TEMPERATURE if temperature is None else temperature
    out_tokens = DEFAULT_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
    in_tokens_limit = DEFAULT_MAX_INPUT_TOKENS if max_input_tokens is None else max_input_tokens
    req_timeout = DEFAULT_TIMEOUT if timeout_seconds is None else timeout_seconds
    retries_limit = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    delay_base = DEFAULT_INITIAL_RETRY_DELAY if initial_delay is None else initial_delay

    start_time = time.perf_counter()

    # 1. Input token budget pre-check
    total_prompt_text = " ".join(m.get("content", "") for m in messages)
    estimated_in_tokens = estimate_tokens(total_prompt_text)

    if estimated_in_tokens > in_tokens_limit:
        CUMULATIVE_TOKEN_USAGE["failed_requests"] += 1
        return {
            "content": "",
            "input_tokens": estimated_in_tokens,
            "output_tokens": 0,
            "total_tokens": estimated_in_tokens,
            "latency_ms": 0.0,
            "retries_attempted": 0,
            "model": selected_model,
            "status": "error",
            "error": f"Token budget exceeded: input has ~{estimated_in_tokens} tokens (max allowed: {in_tokens_limit}).",
        }

    # 2. Client acquisition
    if client is None:
        try:
            client = get_groq_client()
        except ValueError as ve:
            CUMULATIVE_TOKEN_USAGE["failed_requests"] += 1
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0.0,
                "retries_attempted": 0,
                "model": selected_model,
                "status": "error",
                "error": str(ve),
            }

    retries_done = 0
    last_error: Optional[Exception] = None

    for attempt in range(retries_limit + 1):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temp,
                max_tokens=out_tokens,
                timeout=req_timeout,
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            usage = getattr(response, "usage", None)
            actual_in = getattr(usage, "prompt_tokens", estimated_in_tokens) if usage else estimated_in_tokens
            actual_out = getattr(usage, "completion_tokens", 0) if usage else 0
            actual_total = getattr(usage, "total_tokens", actual_in + actual_out) if usage else actual_in

            CUMULATIVE_TOKEN_USAGE["prompt_tokens"] += actual_in
            CUMULATIVE_TOKEN_USAGE["completion_tokens"] += actual_out
            CUMULATIVE_TOKEN_USAGE["total_tokens"] += actual_total
            CUMULATIVE_TOKEN_USAGE["successful_requests"] += 1

            return {
                "content": response.choices[0].message.content or "",
                "input_tokens": actual_in,
                "output_tokens": actual_out,
                "total_tokens": actual_total,
                "latency_ms": elapsed_ms,
                "retries_attempted": retries_done,
                "model": selected_model,
                "status": "success",
                "error": None,
            }

        except groq.AuthenticationError as e:
            # Permanent error: invalid credentials must not be retried
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            CUMULATIVE_TOKEN_USAGE["failed_requests"] += 1
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": elapsed_ms,
                "retries_attempted": retries_done,
                "model": selected_model,
                "status": "error",
                "error": f"Authentication failed: Check your GROQ_API_KEY in .env. ({e})",
            }

        except (groq.NotFoundError, groq.BadRequestError, groq.PermissionDeniedError, groq.UnprocessableEntityError) as e:
            # Permanent error: bad model or payload must not be retried
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            CUMULATIVE_TOKEN_USAGE["failed_requests"] += 1
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": elapsed_ms,
                "retries_attempted": retries_done,
                "model": selected_model,
                "status": "error",
                "error": f"Permanent request error ({type(e).__name__}): {e}",
            }

        except (groq.RateLimitError, groq.InternalServerError, groq.APITimeoutError, groq.APIConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < retries_limit:
                retries_done += 1
                backoff_delay = (delay_base * (2 ** attempt)) + random.uniform(0.1, 0.4)
                err_label = "RateLimit (429)" if isinstance(e, groq.RateLimitError) else type(e).__name__
                print(f"  ⚠️  Temporary error ({err_label}): Retry {retries_done}/{retries_limit} in {backoff_delay:.2f}s...")
                time.sleep(backoff_delay)
            else:
                break
        except Exception as e:
            last_error = e
            break

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    CUMULATIVE_TOKEN_USAGE["failed_requests"] += 1
    return {
        "content": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": elapsed_ms,
        "retries_attempted": retries_done,
        "model": selected_model,
        "status": "error",
        "error": f"Request failed after {retries_done} retries ({type(last_error).__name__}): {last_error}",
    }


def get_token_metrics() -> Dict[str, int]:
    """Return cumulative token usage and request counters."""
    return dict(CUMULATIVE_TOKEN_USAGE)
