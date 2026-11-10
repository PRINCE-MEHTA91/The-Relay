"""Test suite verifying bounded request queue and Server-Sent Events (SSE) streaming."""

import json
import asyncio
import httpx
from fastapi.testclient import TestClient
from app.main import app
from rate_limiter import concurrency_manager, limiter, ConcurrencyQueueManager
from mock_llm import MockMode

client = TestClient(app)


def test_queue_full_and_successful_processing():
    """Verify that bounded queue rejects with HTTP 503 when full and processes queued requests safely."""
    print("=" * 66)
    print("  1. TESTING BOUNDED QUEUE: QUEUE FULL & PROCESSING")
    print("=" * 66)

    # Unit test directly on ConcurrencyQueueManager
    test_queue = ConcurrencyQueueManager(max_concurrency=1, max_queue_size=1, queue_timeout=0.3)

    async def run_queue_tests():
        # Slot 1: Active
        ok1, reason1 = await test_queue.acquire()
        print(f"Request 1 (active): acquired={ok1}, reason={reason1}")
        assert ok1 is True
        assert reason1 == "ACQUIRED"
        assert test_queue.active_count == 1
        assert test_queue.queued_count == 0

        # Slot 2 & Slot 3 simulated concurrently
        async def second_request():
            # Queues up in slot 1
            ok2, reason2 = await test_queue.acquire()
            print(f"Request 2 (queued): acquired={ok2}, reason={reason2}")
            return ok2, reason2

        async def third_request():
            # Queue is already at max_queue_size (1) -> Immediate rejection
            await asyncio.sleep(0.02)
            ok3, reason3 = await test_queue.acquire()
            print(f"Request 3 (queue full): acquired={ok3}, reason={reason3}")
            return ok3, reason3

        task2 = asyncio.create_task(second_request())
        task3 = asyncio.create_task(third_request())

        # Release request 1 after 0.1s so request 2 can succeed before timeout
        await asyncio.sleep(0.1)
        test_queue.release()

        res2, res3 = await asyncio.gather(task2, task3)

        assert res2[0] is True
        assert res2[1] == "ACQUIRED"
        test_queue.release()

        assert res3[0] is False
        assert res3[1] == "QUEUE_FULL"

    asyncio.run(run_queue_tests())
    print("  Queue Full & Processing: PASS ✅\n")


def test_streaming_sse_output():
    """Verify that SSE streaming delivers progressive tokens and final metadata."""
    print("=" * 66)
    print("  2. TESTING SERVER-SENT EVENTS (SSE) STREAMING")
    print("=" * 66)

    payload = {
        "question": "What is the Transformer architecture?",
        "use_mock": True,
        "stream": True,
    }

    # Test dedicated /ask/stream endpoint
    with client.stream("POST", "/ask/stream", json=payload) as response:
        print(f"Response status: {response.status_code}")
        print(f"Content-Type   : {response.headers.get('content-type')}")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = []
        tokens = []
        for line in response.iter_lines():
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                data_json = json.loads(line_str[6:])
                events.append(data_json)
                if data_json.get("token"):
                    tokens.append(data_json["token"])

        print(f"SSE events count : {len(events)}")
        print(f"Tokens received  : {len(tokens)}")
        full_text = "".join(tokens)
        print(f"Streamed preview : \"{full_text[:70]}...\"")

        assert len(events) >= 5
        assert len(tokens) >= 5
        assert events[0].get("type") == "init"
        assert events[-1].get("done") is True
        assert "citations" in events[-1]

    print("  SSE Streaming Output: PASS ✅\n")


def test_client_disconnection():
    """Verify that client disconnection halts streaming and releases concurrency slot."""
    print("=" * 66)
    print("  3. TESTING CLIENT DISCONNECTION AND CLEANUP")
    print("=" * 66)

    initial_active = concurrency_manager.active_count

    payload = {
        "question": "What is Moshi?",
        "use_mock": True,
        "mock_mode": "SLOW",
        "stream": True,
    }

    # Connect and disconnect after reading only the first event
    with client.stream("POST", "/ask/stream", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                print(f"Received first chunk: {line_str[:50]}...")
                # Abruptly exit with-block to simulate client disconnection
                break

    # Give event loop a moment to trigger generator cleanup
    asyncio.run(asyncio.sleep(0.1))

    final_active = concurrency_manager.active_count
    print(f"Active requests before: {initial_active}, after disconnect: {final_active}")
    assert final_active == initial_active
    print("  Client Disconnection Cleanup: PASS ✅\n")


def test_existing_normal_endpoint():
    """Verify that the existing normal response endpoint continues working unchanged."""
    print("=" * 66)
    print("  4. TESTING EXISTING NORMAL ENDPOINT COMPATIBILITY")
    print("=" * 66)

    payload = {
        "question": "What is dynamic chunking?",
        "use_mock": True,
        "stream": False,
    }

    res = client.post("/ask", json=payload)
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"Answer: {data.get('answer')[:70]}...")
    assert res.status_code == 200
    assert "Dynamic chunking" in data.get("answer", "")
    assert data.get("is_valid") is True
    print("  Existing Normal Endpoint: PASS ✅\n")


def main():
    test_queue_full_and_successful_processing()
    test_streaming_sse_output()
    test_client_disconnection()
    test_existing_normal_endpoint()
    print("=" * 66)
    print("  🎉 ALL BOUNDED QUEUE & STREAMING TESTS PASSED!")
    print("=" * 66)


if __name__ == "__main__":
    main()

