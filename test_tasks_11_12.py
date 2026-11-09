"""Test suite verifying Task 11 (FastAPI backend) and Task 12 (Mock LLM Service)."""

import asyncio
import time
from fastapi.testclient import TestClient
from app.main import app
from mock_llm import MockLLMService, MockMode, MockRateLimitError, MockServerError, MockTimeoutError

client = TestClient(app)


def test_fastapi_endpoints():
    """Run all required tests for FastAPI endpoints."""
    print("=" * 66)
    print("  TESTING TASK 11: FASTAPI BACKEND ENDPOINTS")
    print("=" * 66)

    # 1. Test GET /
    res_root = client.get("/")
    print(f"\n1. GET /")
    print(f"   Status : {res_root.status_code}")
    print(f"   Payload: {res_root.json()}")
    assert res_root.status_code == 200
    assert "The Relay" in res_root.json()["message"]
    print("   Result : PASS ✅")

    # 2. Test GET /health
    res_health = client.get("/health")
    print(f"\n2. GET /health")
    print(f"   Status : {res_health.status_code}")
    print(f"   Payload: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["indexed_vectors"] == 215
    print("   Result : PASS ✅")

    # 3. Test POST /ask with valid question (using mock mode for reliable offline testing)
    res_valid = client.post(
        "/ask",
        json={"question": "What is the Transformer architecture?", "use_mock": True},
    )
    print(f"\n3. POST /ask (valid question with mock mode)")
    print(f"   Status : {res_valid.status_code}")
    data = res_valid.json()
    print(f"   Answer : {data['answer']}")
    print(f"   Valid  : {data['is_valid']}")
    print(f"   Citations count: {len(data['citations'])}")
    assert res_valid.status_code == 200
    assert data["is_valid"] is True
    print("   Result : PASS ✅")

    # 4. Test POST /ask with empty question
    res_empty = client.post("/ask", json={"question": "   ", "use_mock": True})
    print(f"\n4. POST /ask (empty question)")
    print(f"   Status : {res_empty.status_code}")
    print(f"   Detail : {res_empty.json()}")
    assert res_empty.status_code == 400
    print("   Result : PASS ✅")

    # 5. Test POST /ask with invalid input (missing question field)
    res_invalid = client.post("/ask", json={"top_k": 5})
    print(f"\n5. POST /ask (invalid input - missing 'question')")
    print(f"   Status : {res_invalid.status_code}")
    print(f"   Detail : {res_invalid.json().get('detail', res_invalid.json())}")
    assert res_invalid.status_code == 422
    print("   Result : PASS ✅")


async def test_mock_llm_modes():
    """Run all required tests for Mock LLM Service modes."""
    print("\n\n" + "=" * 66)
    print("  TESTING TASK 12: MOCK LLM SERVICE MODES")
    print("=" * 66)

    service = MockLLMService(
        base_latency=0.05,
        slow_latency=0.5,
        timeout_duration=0.3,
        overload_penalty_per_request=0.1,
    )
    dummy_messages = [{"role": "user", "content": "Explain Moshi in speech foundation models."}]

    # 1. NORMAL Mode
    t0 = time.perf_counter()
    res_normal = await service.generate_response(dummy_messages, mode=MockMode.NORMAL)
    el_normal = time.perf_counter() - t0
    print(f"\n1. Mode: NORMAL")
    print(f"   Status  : {res_normal['status']}")
    print(f"   Latency : {res_normal['latency_ms']} ms (actual: {el_normal*1000:.1f} ms)")
    print(f"   Content : {res_normal['content'][:75]}...")
    assert res_normal["status"] == "success"
    assert el_normal < 0.3
    print("   Result  : PASS ✅")

    # 2. SLOW Mode
    t0 = time.perf_counter()
    res_slow = await service.generate_response(dummy_messages, mode=MockMode.SLOW)
    el_slow = time.perf_counter() - t0
    print(f"\n2. Mode: SLOW")
    print(f"   Status  : {res_slow['status']}")
    print(f"   Latency : {res_slow['latency_ms']} ms (actual: {el_slow*1000:.1f} ms)")
    assert res_slow["status"] == "success"
    assert el_slow >= 0.45
    print("   Result  : PASS ✅")

    # 3. FAILURE Mode
    res_fail = await service.generate_response(dummy_messages, mode=MockMode.FAILURE)
    print(f"\n3. Mode: FAILURE")
    print(f"   Status     : {res_fail['status']}")
    print(f"   Status Code: {res_fail.get('status_code')}")
    print(f"   Error      : {res_fail['error']}")
    assert res_fail["status"] == "error"
    assert res_fail["status_code"] == 500
    print("   Result     : PASS ✅")

    # 4. RATE_LIMITED Mode
    res_rate = await service.generate_response(dummy_messages, mode=MockMode.RATE_LIMITED)
    print(f"\n4. Mode: RATE_LIMITED")
    print(f"   Status     : {res_rate['status']}")
    print(f"   Status Code: {res_rate.get('status_code')}")
    print(f"   Error      : {res_rate['error']}")
    assert res_rate["status"] == "error"
    assert res_rate["status_code"] == 429
    print("   Result     : PASS ✅")

    # 5. TIMEOUT Mode
    res_timeout = await service.generate_response(dummy_messages, mode=MockMode.TIMEOUT)
    print(f"\n5. Mode: TIMEOUT")
    print(f"   Status     : {res_timeout['status']}")
    print(f"   Status Code: {res_timeout.get('status_code')}")
    print(f"   Error      : {res_timeout['error']}")
    assert res_timeout["status"] == "error"
    assert res_timeout["status_code"] == 504
    print("   Result     : PASS ✅")

    # 6. OVERLOAD Mode (simulate 4 concurrent tasks)
    print(f"\n6. Mode: OVERLOAD (concurrent requests latency ramp)")
    async def call_overload(req_num: int):
        r = await service.generate_response(dummy_messages, mode=MockMode.OVERLOAD)
        return req_num, r["latency_ms"]

    tasks = [call_overload(i) for i in range(4)]
    results = await asyncio.gather(*tasks)
    for req_num, lat in results:
        print(f"   Request #{req_num} latency: {lat:.1f} ms")
    assert all(lat > 50 for _, lat in results)
    print("   Result  : PASS ✅")

    # 7. STREAMING Mode
    print(f"\n7. Mode: STREAMING (token by token generator)")
    tokens_received = []
    async for token in service.stream_response(dummy_messages, chunk_delay=0.01):
        tokens_received.append(token)
    streamed_text = "".join(tokens_received)
    print(f"   Tokens streamed : {len(tokens_received)}")
    print(f"   Streamed text   : \"{streamed_text[:75]}...\"")
    assert len(tokens_received) > 5
    assert "Moshi" in streamed_text
    print("   Result          : PASS ✅")


def main():
    test_fastapi_endpoints()
    asyncio.run(test_mock_llm_modes())
    print("\n" + "=" * 66)
    print("  🎉 ALL FASTAPI & MOCK LLM TESTS PASSED!")
    print("=" * 66)


if __name__ == "__main__":
    main()

