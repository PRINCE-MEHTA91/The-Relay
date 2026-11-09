"""Tests for SlidingWindowRateLimiter and FastAPI rate-limiting integration."""

import asyncio
from fastapi.testclient import TestClient
from app.main import app
from rate_limiter import SlidingWindowRateLimiter, limiter

client = TestClient(app)


def test_rate_limiter_unit():
    """Unit tests for sliding window rate limiter logic."""
    print("=" * 66)
    print("  TESTING RATE LIMITER (UNIT TESTS)")
    print("=" * 66)

    test_limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=2.0)

    async def run_unit_tests():
        # Requests 1, 2, 3 should succeed
        for i in range(1, 4):
            allowed, remaining, _ = await test_limiter.check_rate_limit("client_a")
            print(f"Request {i}: allowed={allowed}, remaining={remaining}")
            assert allowed is True
            assert remaining == 3 - i

        # Request 4 should be rejected with 429 and retry_after
        allowed, remaining, retry_after = await test_limiter.check_rate_limit("client_a")
        print(f"Request 4: allowed={allowed}, remaining={remaining}, retry_after={retry_after}s")
        assert allowed is False
        assert retry_after > 0

        # Different client IP should have full quota
        allowed_b, rem_b, _ = await test_limiter.check_rate_limit("client_b")
        print(f"Client B Request 1: allowed={allowed_b}, remaining={rem_b}")
        assert allowed_b is True
        assert rem_b == 2

    asyncio.run(run_unit_tests())
    print("  Result: PASS ✅\n")


def test_fastapi_rate_limit_integration():
    """Integration test verifying HTTP 429 response and Retry-After header on /ask."""
    print("=" * 66)
    print("  TESTING FASTAPI HTTP 429 INTEGRATION")
    print("=" * 66)

    # Set small limit for testing
    limiter.reset()
    limiter.max_requests = 3
    limiter.window_seconds = 10.0

    payload = {
        "question": "What is the Transformer architecture?",
        "use_mock": True,
    }

    # 1. First 3 requests should be accepted (HTTP 200)
    for i in range(1, 4):
        res = client.post("/ask", json=payload, headers={"X-Forwarded-For": "192.168.1.100"})
        print(f"Request {i}: status={res.status_code}, remaining={res.headers.get('X-RateLimit-Remaining')}")
        assert res.status_code == 200

    # 2. 4th request must be rejected with HTTP 429
    res_rejected = client.post("/ask", json=payload, headers={"X-Forwarded-For": "192.168.1.100"})
    print(f"\nRequest 4 (Burst Exceeded):")
    print(f"  Status code : {res_rejected.status_code}")
    print(f"  Retry-After : {res_rejected.headers.get('Retry-After')} seconds")
    print(f"  Error Detail: {res_rejected.json().get('detail')}")

    assert res_rejected.status_code == 429
    assert "Retry-After" in res_rejected.headers
    assert "Rate limit exceeded" in res_rejected.json()["detail"]
    print("  Result: PASS ✅\n")

    # Reset limiter to default
    limiter.reset()
    limiter.max_requests = 10
    limiter.window_seconds = 60.0


def main():
    test_rate_limiter_unit()
    test_fastapi_rate_limit_integration()
    print("=" * 66)
    print("  🎉 ALL RATE LIMITER TESTS PASSED!")
    print("=" * 66)


if __name__ == "__main__":
    main()
