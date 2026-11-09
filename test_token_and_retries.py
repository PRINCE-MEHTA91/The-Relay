"""Test suite verifying token budget enforcement, timeout handling, and exponential retries."""

import time
from typing import Dict, Any, List
from unittest.mock import MagicMock
import groq
from groq_service import send_chat_completion, estimate_tokens, get_token_metrics


class DummyResponse:
    """Mock Groq ChatCompletion response."""
    def __init__(self, content: str = "Test response", prompt_tokens: int = 25, completion_tokens: int = 15):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


def test_token_limit_handling():
    """Verify that requests exceeding input token budget are rejected pre-flight."""
    print("=" * 66)
    print("  1. TESTING TOKEN BUDGET MANAGEMENT")
    print("=" * 66)

    # 1. Normal prompt under budget
    normal_prompt = [{"role": "user", "content": "What is self-attention?"}]
    est = estimate_tokens(normal_prompt[0]["content"])
    print(f"Normal prompt (~{est} tokens) with budget 4096: ALLOWED")
    assert est < 4096

    # 2. Huge prompt exceeding strict budget (budget = 30 tokens)
    huge_text = "word " * 100  # ~130 tokens
    huge_prompt = [{"role": "user", "content": huge_text}]

    mock_client = MagicMock()
    res = send_chat_completion(
        messages=huge_prompt,
        max_input_tokens=30,
        client=mock_client,
    )

    print(f"Excessive prompt (~{res['input_tokens']} tokens) with budget 30:")
    print(f"  Status : {res['status']}")
    print(f"  Error  : {res['error']}")
    assert res["status"] == "error"
    assert "Token budget exceeded" in res["error"]
    assert not mock_client.chat.completions.create.called
    print("  Result : PASS ✅\n")


def test_retry_success():
    """Verify that a temporary failure on attempt 1 succeeds on attempt 2."""
    print("=" * 66)
    print("  2. TESTING RETRY SUCCESS ON TEMPORARY FAILURE")
    print("=" * 66)

    mock_client = MagicMock()
    # Attempt 1: Rate limit 429; Attempt 2: Success
    mock_client.chat.completions.create.side_effect = [
        groq.RateLimitError("Rate limit 429", response=MagicMock(status_code=429), body=None),
        DummyResponse("Recovered after retry", prompt_tokens=30, completion_tokens=10),
    ]

    messages = [{"role": "user", "content": "Hello"}]
    res = send_chat_completion(
        messages=messages,
        max_retries=2,
        initial_delay=0.1,  # fast backoff for tests
        client=mock_client,
    )

    print(f"Response status    : {res['status']}")
    print(f"Retries attempted  : {res['retries_attempted']}")
    print(f"Tokens tracked     : {res['input_tokens']} in / {res['output_tokens']} out")
    print(f"Content            : \"{res['content']}\"")

    assert res["status"] == "success"
    assert res["retries_attempted"] == 1
    assert res["input_tokens"] == 30
    assert res["output_tokens"] == 10
    print("  Result : PASS ✅\n")


def test_retry_exhaustion():
    """Verify that persistent temporary failures stop after max_retries."""
    print("=" * 66)
    print("  3. TESTING RETRY EXHAUSTION")
    print("=" * 66)

    mock_client = MagicMock()
    # Always fail with InternalServerError (500)
    mock_client.chat.completions.create.side_effect = groq.InternalServerError(
        "Server 500 error", response=MagicMock(status_code=500), body=None
    )

    messages = [{"role": "user", "content": "Hello"}]
    res = send_chat_completion(
        messages=messages,
        max_retries=2,
        initial_delay=0.1,
        client=mock_client,
    )

    print(f"Response status    : {res['status']}")
    print(f"Retries attempted  : {res['retries_attempted']}")
    print(f"Error message      : {res['error']}")

    assert res["status"] == "error"
    assert res["retries_attempted"] == 2
    assert "Request failed after 2 retries" in res["error"]
    print("  Result : PASS ✅\n")


def test_permanent_error_not_retried():
    """Verify that permanent errors (e.g. 401 AuthenticationError) fail immediately without retrying."""
    print("=" * 66)
    print("  4. TESTING PERMANENT ERROR (NO RETRIES)")
    print("=" * 66)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.AuthenticationError(
        "Invalid API Key", response=MagicMock(status_code=401), body=None
    )

    messages = [{"role": "user", "content": "Hello"}]
    res = send_chat_completion(
        messages=messages,
        max_retries=3,
        client=mock_client,
    )

    print(f"Response status    : {res['status']}")
    print(f"Retries attempted  : {res['retries_attempted']}")
    print(f"Error message      : {res['error']}")

    assert res["status"] == "error"
    assert res["retries_attempted"] == 0  # Zero retries wasted on bad credentials
    assert "Authentication failed" in res["error"]
    print("  Result : PASS ✅\n")


def test_timeout_handling():
    """Verify that timeout errors trigger exponential retries."""
    print("=" * 66)
    print("  5. TESTING TIMEOUT HANDLING")
    print("=" * 66)

    mock_client = MagicMock()
    # Attempt 1: Timeout; Attempt 2: Success
    mock_client.chat.completions.create.side_effect = [
        groq.APITimeoutError(request=MagicMock()),
        DummyResponse("Recovered after timeout", prompt_tokens=20, completion_tokens=8),
    ]

    messages = [{"role": "user", "content": "Hello"}]
    res = send_chat_completion(
        messages=messages,
        max_retries=2,
        initial_delay=0.1,
        client=mock_client,
    )

    print(f"Response status    : {res['status']}")
    print(f"Retries attempted  : {res['retries_attempted']}")
    print(f"Content            : \"{res['content']}\"")

    assert res["status"] == "success"
    assert res["retries_attempted"] == 1
    print("  Result : PASS ✅\n")


def main():
    test_token_limit_handling()
    test_retry_success()
    test_retry_exhaustion()
    test_permanent_error_not_retried()
    test_timeout_handling()

    metrics = get_token_metrics()
    print("=" * 66)
    print("  📊 CUMULATIVE TOKEN AUDIT METRICS")
    print(f"  Prompt tokens     : {metrics['prompt_tokens']}")
    print(f"  Completion tokens : {metrics['completion_tokens']}")
    print(f"  Total tokens      : {metrics['total_tokens']}")
    print(f"  Successful calls  : {metrics['successful_requests']}")
    print(f"  Failed calls      : {metrics['failed_requests']}")
    print("=" * 66)
    print("  🎉 ALL RESILIENCE & BUDGET TESTS PASSED!")
    print("=" * 66)


if __name__ == "__main__":
    main()

