"""Test suite verifying all five load scenarios and CSV schema validation."""

import os
import sys
import csv
import asyncio
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
try:
    from experiments.load_tests.load_test import (
        CSV_FIELDNAMES,
        VALID_SOURCES,
        VALID_STATUSES,
        VALID_ANSWER_GRADES,
        run_steady_load,
        run_burst_load,
        run_overload_load,
        run_limit_halved_load,
        run_budget_exhausted_load,
        save_results_to_csv,
    )
except ImportError:
    from load_test import (
        CSV_FIELDNAMES,
        VALID_SOURCES,
        VALID_STATUSES,
        VALID_ANSWER_GRADES,
        run_steady_load,
        run_burst_load,
        run_overload_load,
        run_limit_halved_load,
        run_budget_exhausted_load,
        save_results_to_csv,
    )

TEST_CSV_FILE = "test_results_audit.csv"


def validate_csv_schema(filepath: str):
    """Validate that CSV adheres strictly to assignment columns and allowed enum values."""
    assert os.path.exists(filepath), f"File {filepath} was not created."
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDNAMES, f"Fieldnames mismatch: {reader.fieldnames} vs {CSV_FIELDNAMES}"
        rows = list(reader)
        assert len(rows) > 0, "CSV has no rows."

        for idx, row in enumerate(rows, start=1):
            assert row["source"] in VALID_SOURCES, f"Row {idx} invalid source: {row['source']}"
            assert row["status"] in VALID_STATUSES, f"Row {idx} invalid status: {row['status']}"
            assert row["answer_grade"] in VALID_ANSWER_GRADES, f"Row {idx} invalid grade: {row['answer_grade']}"
            assert row["request_id"].startswith("req_"), f"Row {idx} invalid req_id: {row['request_id']}"
            assert float(row["total_ms"]) >= 0.0, f"Row {idx} negative total_ms"


def test_all_five_load_scenarios():
    """Execute all five load scenarios in-process using stand-in and validate outputs."""
    print("=" * 66)
    print("  TESTING PRIORITY 1: ALL FIVE LOAD SCENARIOS & CSV AUDIT")
    print("=" * 66)

    all_records = []

    async def run_tests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Steady Load
            print("\n1. Testing Scenario: Steady Load")
            res_steady = await run_steady_load(
                client=client,
                base_url="http://test",
                source="stand-in",
                request_count=3,
                concurrency=1,
            )
            assert len(res_steady) == 3
            assert all(r["status"] == "ok" for r in res_steady)
            assert all(r["source"] == "stand-in" for r in res_steady)
            all_records.extend(res_steady)
            print("   Steady Load: PASS ✅")

            # 2. Burst Load
            print("\n2. Testing Scenario: Burst Load")
            res_burst = await run_burst_load(
                client=client,
                base_url="http://test",
                source="stand-in",
                request_count=8,
                concurrency=8,
            )
            assert len(res_burst) == 8
            statuses = {r["status"] for r in res_burst}
            print(f"   Burst outcomes: {statuses}")
            assert "ok" in statuses
            all_records.extend(res_burst)
            print("   Burst Load: PASS ✅")

            # 3. Sustained Overload
            print("\n3. Testing Scenario: Sustained Overload")
            res_overload = await run_overload_load(
                client=client,
                base_url="http://test",
                source="stand-in",
                request_count=6,
                concurrency=4,
                duration_seconds=3.0,
            )
            assert len(res_overload) > 0
            all_records.extend(res_overload)
            print("   Sustained Overload: PASS ✅")

            # 4. Limit Halved
            print("\n4. Testing Scenario: Per-Minute Limit Halved")
            res_halved = await run_limit_halved_load(
                client=client,
                base_url="http://test",
                source="stand-in",
                request_count=8,
            )
            assert len(res_halved) == 8
            halved_statuses = [r["status"] for r in res_halved]
            print(f"   Halved outcomes: {halved_statuses}")
            assert "rate_limited" in halved_statuses
            all_records.extend(res_halved)
            print("   Limit Halved: PASS ✅")

            # 5. Budget Exhausted
            print("\n5. Testing Scenario: Daily Budget Exhausted (Stand-in)")
            res_exhausted = await run_budget_exhausted_load(
                client=client,
                base_url="http://test",
                request_count=4,
            )
            assert len(res_exhausted) == 4
            assert all(r["status"] == "rate_limited" for r in res_exhausted)
            assert all(r["source"] == "stand-in" for r in res_exhausted)
            all_records.extend(res_exhausted)
            print("   Budget Exhausted: PASS ✅")

    asyncio.run(run_tests())

    # 6. Save to CSV and validate schema
    print("\n6. Validating CSV Schema and Enums")
    save_results_to_csv(all_records, filepath=TEST_CSV_FILE, append=False)
    validate_csv_schema(TEST_CSV_FILE)
    print(f"   CSV Schema Validation: PASS ✅ ({len(all_records)} verified records)")

    # Clean up test artifact
    if os.path.exists(TEST_CSV_FILE):
        os.remove(TEST_CSV_FILE)

    print("\n" + "=" * 66)
    print("  🎉 ALL FIVE LOAD SCENARIOS & CSV AUDIT TESTS PASSED!")
    print("=" * 66)


if __name__ == "__main__":
    test_all_five_load_scenarios()

