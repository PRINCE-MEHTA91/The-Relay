"""Load testing script for The Relay evaluating five required load scenarios."""

import os
import sys
import time
import json
import uuid
import csv
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import httpx

from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from core.rate_limiter import limiter
except ImportError:
    from rate_limiter import limiter

# Assignment-specified exact CSV column headers
CSV_FIELDNAMES = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]

# Allowed values per assignment requirements
VALID_SOURCES = {"real", "stand-in"}
VALID_STATUSES = {"ok", "rate_limited", "timeout", "error"}
VALID_ANSWER_GRADES = {"", "correct", "wrong", "refused_correctly", "refused_wrongly"}

# Representative sample questions for load generation
TEST_QUESTIONS = [
    "What is the Transformer architecture?",
    "How does self-attention work?",
    "What is Moshi?",
    "Explain dynamic chunking.",
    "What are the key advantages of multi-head attention?",
    "How does Moshi handle spoken dialogue?",
    "Compare dynamic chunking with fixed-size chunking.",
]


async def execute_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    scenario: str,
    source: str = "stand-in",
    question: str = "What is the Transformer architecture?",
    top_k: int = 5,
    stream: bool = False,
    mock_mode: Optional[str] = None,
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Execute a single query against /ask and record metrics adhering to schema."""
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    start_time_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    first_token_ms: Any = ""
    total_ms: Any = ""
    status = "error"
    retries: Any = ""
    tokens_in: Any = ""
    tokens_out: Any = ""
    retrieval_setting = f"top_k={top_k}"
    answer_grade = ""

    payload: Dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "use_mock": (source == "stand-in"),
        "stream": stream,
    }
    if source == "stand-in" and mock_mode:
        payload["mock_mode"] = mock_mode

    url = f"{base_url.rstrip('/')}/ask"

    if stream:
        try:
            async with client.stream("POST", url, json=payload, timeout=timeout_seconds) as response:
                http_code = response.status_code
                if http_code == 200:
                    status = "ok"
                    async for line in response.aiter_lines():
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            try:
                                chunk = json.loads(line_str[6:])
                                if first_token_ms == "" and chunk.get("token"):
                                    first_token_ms = round((time.perf_counter() - t0) * 1000, 2)
                                if chunk.get("done"):
                                    if "input_tokens" in chunk:
                                        tokens_in = chunk["input_tokens"]
                                    if "output_tokens" in chunk:
                                        tokens_out = chunk["output_tokens"]
                                    if "retries_attempted" in chunk:
                                        retries = chunk["retries_attempted"]
                            except json.JSONDecodeError:
                                pass
                elif http_code == 429:
                    status = "rate_limited"
                elif http_code == 504:
                    status = "timeout"
                else:
                    status = "error"
                total_ms = round((time.perf_counter() - t0) * 1000, 2)
        except httpx.TimeoutException:
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            status = "timeout"
        except Exception:
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            status = "error"
    else:
        try:
            response = await client.post(url, json=payload, timeout=timeout_seconds)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            http_code = response.status_code
            if http_code == 200:
                status = "ok"
                try:
                    data = response.json()
                    tokens_in = data.get("input_tokens", "")
                    tokens_out = data.get("output_tokens", "")
                    retries = data.get("retries_attempted", 0)
                except Exception:
                    pass
            elif http_code == 429:
                status = "rate_limited"
            elif http_code == 504:
                status = "timeout"
            else:
                status = "error"
        except httpx.TimeoutException:
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            status = "timeout"
        except Exception:
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            status = "error"

    return {
        "request_id": req_id,
        "scenario": scenario,
        "source": source,
        "start_time": start_time_iso,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": retries,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retrieval_setting": retrieval_setting,
        "answer_grade": answer_grade,
    }


async def run_steady_load(
    client: httpx.AsyncClient,
    base_url: str,
    source: str = "stand-in",
    request_count: int = 6,
    concurrency: int = 2,
    stream: bool = False,
    mock_mode: Optional[str] = None,
    interval_seconds: float = 0.5,
) -> List[Dict[str, Any]]:
    """Scenario 1: Send paced requests with low concurrency to test baseline performance."""
    print(f"\n--- Scenario 1: Steady Load (Requests: {request_count}, Concurrency: {concurrency}, Source: {source}) ---")
    results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(idx: int):
        async with semaphore:
            q = TEST_QUESTIONS[idx % len(TEST_QUESTIONS)]
            res = await execute_single_request(
                client=client,
                base_url=base_url,
                scenario="steady",
                source=source,
                question=q,
                stream=stream,
                mock_mode=mock_mode,
            )
            print(f"  [Steady {idx+1}/{request_count}] Status: {res['status']}, Latency: {res['total_ms']}ms")
            results.append(res)
            await asyncio.sleep(interval_seconds)

    tasks = [asyncio.create_task(worker(i)) for i in range(request_count)]
    await asyncio.gather(*tasks)
    return results


async def run_burst_load(
    client: httpx.AsyncClient,
    base_url: str,
    source: str = "stand-in",
    request_count: int = 10,
    concurrency: int = 10,
    stream: bool = False,
    mock_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scenario 2: Send simultaneous requests to test queue capacity and backpressure."""
    print(f"\n--- Scenario 2: Burst Load (Requests: {request_count}, Concurrency: {concurrency}, Source: {source}) ---")
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(idx: int):
        async with semaphore:
            q = TEST_QUESTIONS[idx % len(TEST_QUESTIONS)]
            res = await execute_single_request(
                client=client,
                base_url=base_url,
                scenario="burst",
                source=source,
                question=q,
                stream=stream,
                mock_mode=mock_mode,
            )
            print(f"  [Burst {idx+1}/{request_count}] Status: {res['status']}, Latency: {res['total_ms']}ms")
            return res

    tasks = [worker(i) for i in range(request_count)]
    results = await asyncio.gather(*tasks)
    return list(results)


async def run_overload_load(
    client: httpx.AsyncClient,
    base_url: str,
    source: str = "stand-in",
    request_count: int = 15,
    concurrency: int = 8,
    duration_seconds: Optional[float] = None,
    stream: bool = False,
    mock_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scenario 3: Send high concurrent requests to test rate limiting and rejection."""
    print(f"\n--- Scenario 3: Sustained Overload (Requests: {request_count}, Concurrency: {concurrency}, Source: {source}) ---")
    results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.perf_counter()

    async def worker(idx: int):
        if duration_seconds and (time.perf_counter() - start_time) > duration_seconds:
            return
        async with semaphore:
            q = TEST_QUESTIONS[idx % len(TEST_QUESTIONS)]
            res = await execute_single_request(
                client=client,
                base_url=base_url,
                scenario="sustained_overload",
                source=source,
                question=q,
                stream=stream,
                mock_mode=mock_mode,
            )
            print(f"  [Overload {idx+1}/{request_count}] Status: {res['status']}, Latency: {res['total_ms']}ms")
            results.append(res)

    tasks = [asyncio.create_task(worker(i)) for i in range(request_count)]
    await asyncio.gather(*tasks)
    return results


async def run_limit_halved_load(
    client: httpx.AsyncClient,
    base_url: str,
    source: str = "stand-in",
    request_count: int = 8,
    concurrency: int = 2,
    stream: bool = False,
    mock_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scenario 4: Test halved per-minute quota to observe early rate-limiting."""
    print(f"\n--- Scenario 4: Per-Minute Limit Halved (Requests: {request_count}, Source: {source}) ---")
    results: List[Dict[str, Any]] = []

    # Temporarily halve rate limiter capacity (from 10 to 5)
    original_limit = limiter.max_requests
    limiter.max_requests = max(2, original_limit // 2)
    limiter.reset()

    try:
        for idx in range(request_count):
            q = TEST_QUESTIONS[idx % len(TEST_QUESTIONS)]
            res = await execute_single_request(
                client=client,
                base_url=base_url,
                scenario="limit_halved",
                source=source,
                question=q,
                stream=stream,
                mock_mode=mock_mode,
            )
            print(f"  [Halved {idx+1}/{request_count}] Status: {res['status']}, Latency: {res['total_ms']}ms")
            results.append(res)
            await asyncio.sleep(0.05)
    finally:
        limiter.max_requests = original_limit
        limiter.reset()

    return results


async def run_budget_exhausted_load(
    client: httpx.AsyncClient,
    base_url: str,
    request_count: int = 5,
    concurrency: int = 1,
    stream: bool = False,
) -> List[Dict[str, Any]]:
    """Scenario 5: Simulate daily quota exhaustion using stand-in without touching real Groq."""
    print(f"\n--- Scenario 5: Daily Budget Exhausted (Requests: {request_count}, Source: stand-in) ---")
    results: List[Dict[str, Any]] = []

    for idx in range(request_count):
        q = TEST_QUESTIONS[idx % len(TEST_QUESTIONS)]
        res = await execute_single_request(
            client=client,
            base_url=base_url,
            scenario="budget_exhausted",
            source="stand-in",
            question=q,
            stream=stream,
            mock_mode="RATE_LIMITED",
        )
        print(f"  [Exhausted {idx+1}/{request_count}] Status: {res['status']} (Simulated 429 Daily Exhaustion)")
        results.append(res)
        await asyncio.sleep(0.05)

    return results


def calculate_percentiles(values: List[float]) -> Tuple[float, float, float]:
    """Calculate p50, p95, and p99 percentiles."""
    if not values:
        return 0.0, 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50 = sorted_vals[int(n * 0.50)]
    p95 = sorted_vals[min(int(n * 0.95), n - 1)]
    p99 = sorted_vals[min(int(n * 0.99), n - 1)]
    return round(p50, 1), round(p95, 1), round(p99, 1)


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str = "results.csv", append: bool = False):
    """Save test results into results.csv validating schema compliance."""
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    mode = "a" if append else "w"
    with open(filepath, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not append or not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDNAMES})
    print(f"\n📁 Saved {len(results)} records to {filepath}")


def print_summary_table(results: List[Dict[str, Any]], duration_seconds: float = 0.0):
    """Print an aggregated statistical summary of test results."""
    total = len(results)
    if total == 0:
        print("No test results to summarize.")
        return

    ok_cnt = sum(1 for r in results if r["status"] == "ok")
    rate_limited_cnt = sum(1 for r in results if r["status"] == "rate_limited")
    timeout_cnt = sum(1 for r in results if r["status"] == "timeout")
    error_cnt = sum(1 for r in results if r["status"] == "error")

    latencies = [float(r["total_ms"]) for r in results if isinstance(r["total_ms"], (int, float))]
    first_tokens = [float(r["first_token_ms"]) for r in results if isinstance(r["first_token_ms"], (int, float))]

    p50_lat, p95_lat, p99_lat = calculate_percentiles(latencies)
    p50_ttft, p95_ttft, p99_ttft = calculate_percentiles(first_tokens)
    offered_rate = round(total / max(0.001, duration_seconds), 1) if duration_seconds > 0 else "N/A"

    print("\n" + "=" * 66)
    print("                     LOAD TEST SUMMARY")
    print("=" * 66)
    print(f"Total Requests      : {total}")
    print(f"Duration            : {round(duration_seconds, 2)} s")
    print(f"Offered Rate        : {offered_rate} req/s")
    print(f"Status 'ok' (200)   : {ok_cnt} ({round(ok_cnt/total*100, 1)}%)")
    print(f"Rate Limited (429)  : {rate_limited_cnt} ({round(rate_limited_cnt/total*100, 1)}%)")
    print(f"Timeouts (504)      : {timeout_cnt} ({round(timeout_cnt/total*100, 1)}%)")
    print(f"Errors (500/503)    : {error_cnt} ({round(error_cnt/total*100, 1)}%)")
    print(f"Total Latency (ms)  : p50={p50_lat} | p95={p95_lat} | p99={p99_lat}")
    print(f"First Token (ms)    : p50={p50_ttft} | p95={p95_ttft} | p99={p99_ttft}")
    print("=" * 66)


async def main():
    """Main CLI entry point for executing configurable load testing scenarios."""
    parser = argparse.ArgumentParser(description="The Relay Load Testing Harness")
    parser.add_argument(
        "--scenario",
        choices=["steady", "burst", "overload", "limit_halved", "budget_exhausted", "all"],
        default="steady",
        help="Load scenario to execute (default: steady).",
    )
    parser.add_argument(
        "--source",
        choices=["stand-in", "real"],
        default="stand-in",
        help="LLM provider: 'stand-in' (default, quota safe) or 'real'.",
    )
    parser.add_argument(
        "--mock-mode",
        choices=["NORMAL", "SLOW", "FAILURE", "RATE_LIMITED", "TIMEOUT", "OVERLOAD"],
        default="NORMAL",
        help="Operational mode for mock LLM when --source is stand-in.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total requests to dispatch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrency limit.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Max duration in seconds for sustained overload.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use Server-Sent Events (SSE) streaming to record first_token_ms.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Target base URL of FastAPI server (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run load test in-process via httpx ASGITransport without external server.",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV file path (default: results.csv).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing CSV file instead of overwriting.",
    )

    args = parser.parse_args()

    if args.source == "real":
        print("\n⚠️  CAUTION: Testing with real Groq LLM integration.")
        print("   Keep request counts modest to avoid exhausting the daily rate limit quota.")

    if args.in_process:
        from app.main import app
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        effective_base_url = "http://test"
    else:
        client = httpx.AsyncClient(base_url=args.base_url)
        effective_base_url = args.base_url

    all_results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    async with client:
        scenarios_to_run = (
            ["steady", "burst", "overload", "limit_halved", "budget_exhausted"]
            if args.scenario == "all"
            else [args.scenario]
        )

        for scn in scenarios_to_run:
            if scn == "steady":
                reqs = args.requests or (3 if args.source == "real" else 6)
                conc = args.concurrency or 2
                res = await run_steady_load(
                    client=client,
                    base_url=effective_base_url,
                    source=args.source,
                    request_count=reqs,
                    concurrency=conc,
                    stream=args.stream,
                    mock_mode=args.mock_mode,
                )
                all_results.extend(res)

            elif scn == "burst":
                reqs = args.requests or (5 if args.source == "real" else 10)
                conc = args.concurrency or (5 if args.source == "real" else 10)
                res = await run_burst_load(
                    client=client,
                    base_url=effective_base_url,
                    source=args.source,
                    request_count=reqs,
                    concurrency=conc,
                    stream=args.stream,
                    mock_mode=args.mock_mode,
                )
                all_results.extend(res)

            elif scn == "overload":
                reqs = args.requests or (6 if args.source == "real" else 15)
                conc = args.concurrency or (4 if args.source == "real" else 8)
                res = await run_overload_load(
                    client=client,
                    base_url=effective_base_url,
                    source=args.source,
                    request_count=reqs,
                    concurrency=conc,
                    duration_seconds=args.duration or 5.0,
                    stream=args.stream,
                    mock_mode=args.mock_mode,
                )
                all_results.extend(res)

            elif scn == "limit_halved":
                reqs = args.requests or (4 if args.source == "real" else 8)
                res = await run_limit_halved_load(
                    client=client,
                    base_url=effective_base_url,
                    source=args.source,
                    request_count=reqs,
                    stream=args.stream,
                    mock_mode=args.mock_mode,
                )
                all_results.extend(res)

            elif scn == "budget_exhausted":
                reqs = args.requests or 5
                res = await run_budget_exhausted_load(
                    client=client,
                    base_url=effective_base_url,
                    request_count=reqs,
                    stream=args.stream,
                )
                all_results.extend(res)

    total_duration = time.perf_counter() - t_start
    save_results_to_csv(all_results, filepath=args.output, append=args.append)
    print_summary_table(all_results, duration_seconds=total_duration)


if __name__ == "__main__":
    asyncio.run(main())
