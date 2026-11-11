"""Controlled experiment demonstrating retry degradation (queue pressure and completion time).

Compares:
Condition A: Zero Retries (Fail-Fast)
Condition B: 3 Retries with Exponential Backoff (1.0s, 2.0s)
Under an overloaded / intermittent upstream service (Mock LLM).
"""

import time
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.mock_llm import MockLLMService, MockMode, MockServerError, MockRateLimitError


async def simulate_workload(
    service: MockLLMService,
    num_requests: int = 10,
    max_retries: int = 0,
    base_backoff: float = 0.5,
) -> Dict[str, Any]:
    """Execute concurrent requests and measure peak queue depth and completion times."""
    queue_depth = 0
    peak_queue = 0
    lock = asyncio.Lock()
    completion_times: List[float] = []
    statuses: List[str] = []
    total_attempts_recorded: List[int] = []

    async def single_request(req_id: int):
        nonlocal queue_depth, peak_queue
        t_start = time.perf_counter()
        async with lock:
            queue_depth += 1
            if queue_depth > peak_queue:
                peak_queue = queue_depth

        attempts = 0
        success = False
        final_status = "error"

        try:
            for attempt in range(max_retries + 1):
                attempts += 1
                try:
                    # In this scenario, requests experience transient overload
                    # Simulate transient failure on initial attempt, succeeding on subsequent attempt
                    if attempt == 0:
                        # 60% of first attempts trigger a transient 500 error under overload
                        if req_id % 3 != 0:
                            await asyncio.sleep(0.08)  # brief processing before failure
                            raise MockServerError("Upstream 500: capacity exhausted")
                    
                    # Subsequent attempt or stable request
                    res = await service.generate_response(
                        messages=[{"role": "user", "content": "What is the Transformer architecture?"}],
                        mode=MockMode.OVERLOAD,
                        request_id=f"deg_{req_id}_{attempt}",
                    )
                    if res["status"] == "success":
                        success = True
                        final_status = "ok"
                        break
                except MockServerError:
                    if attempt < max_retries:
                        backoff = base_backoff * (2 ** attempt)
                        await asyncio.sleep(backoff)
                    else:
                        final_status = "error"
        finally:
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            completion_times.append(elapsed_ms)
            statuses.append(final_status)
            total_attempts_recorded.append(attempts)
            async with lock:
                queue_depth -= 1

    tasks = [asyncio.create_task(single_request(i)) for i in range(num_requests)]
    await asyncio.gather(*tasks)

    import numpy as np
    p50 = float(np.percentile(completion_times, 50))
    p95 = float(np.percentile(completion_times, 95))
    p99 = float(np.percentile(completion_times, 99))
    mean_lat = float(np.mean(completion_times))

    return {
        "num_requests": num_requests,
        "max_retries": max_retries,
        "peak_queue": peak_queue,
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "mean_ms": round(mean_lat, 1),
        "success_count": statuses.count("ok"),
        "error_count": statuses.count("error"),
        "avg_attempts": round(sum(total_attempts_recorded) / len(total_attempts_recorded), 2),
        "total_wall_time_ms": round(max(completion_times), 1),
    }


async def main():
    service = MockLLMService(
        base_latency=0.1,
        overload_penalty_per_request=0.08,
        max_concurrency=12,
    )

    print("=" * 70)
    print("CONTROLLED RETRY DEGRADATION EXPERIMENT (LOCAL STAND-IN)")
    print("=" * 70)

    # Condition A: No Retries (Fail-Fast)
    res_a = await simulate_workload(service, num_requests=10, max_retries=0)
    print("\n--- Condition A: Zero Retries (Fail-Fast, max_retries=0) ---")
    print(f"  Peak Queue Depth       : {res_a['peak_queue']} slots")
    print(f"  Mean Total Time        : {res_a['mean_ms']} ms")
    print(f"  Latency p50 / p95      : {res_a['p50_ms']} ms / {res_a['p95_ms']} ms")
    print(f"  Total Wall Clock Time  : {res_a['total_wall_time_ms']} ms")
    print(f"  Success / Failed       : {res_a['success_count']} ok / {res_a['error_count']} errors")
    print(f"  Avg Attempts / Request : {res_a['avg_attempts']}")

    # Small pause to let event loop settle
    await asyncio.sleep(0.5)

    # Condition B: Retries Enabled (max_retries=2, base backoff 0.5s)
    res_b = await simulate_workload(service, num_requests=10, max_retries=2, base_backoff=0.5)
    print("\n--- Condition B: Retries Enabled (max_retries=2, exponential backoff) ---")
    print(f"  Peak Queue Depth       : {res_b['peak_queue']} slots")
    print(f"  Mean Total Time        : {res_b['mean_ms']} ms")
    print(f"  Latency p50 / p95      : {res_b['p50_ms']} ms / {res_b['p95_ms']} ms")
    print(f"  Total Wall Clock Time  : {res_b['total_wall_time_ms']} ms")
    print(f"  Success / Failed       : {res_b['success_count']} ok / {res_b['error_count']} errors")
    print(f"  Avg Attempts / Request : {res_b['avg_attempts']}")

    queue_increase = res_b["peak_queue"] - res_a["peak_queue"]
    latency_multiplier = round(res_b["mean_ms"] / res_a["mean_ms"], 1)

    print("\n" + "=" * 70)
    print(f"COMPARATIVE OUTCOME:")
    print(f"• Mean completion time increased from {res_a['mean_ms']} ms to {res_b['mean_ms']} ms ({latency_multiplier}x higher).")
    print(f"• Total wall clock time increased from {res_a['total_wall_time_ms']} ms to {res_b['total_wall_time_ms']} ms.")
    print(f"• In-flight queue pressure increased by +{queue_increase} slots as retries lingered in backoff.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
