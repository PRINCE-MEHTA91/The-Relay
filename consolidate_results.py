"""Integration module validating and consolidating all benchmark outputs into results.csv."""

import os
import csv
from typing import Dict, Any, List

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

VALID_SOURCES = {"real", "stand-in"}
VALID_STATUSES = {"ok", "rate_limited", "timeout", "error"}
VALID_ANSWER_GRADES = {"", "correct", "wrong", "refused_correctly", "refused_wrongly"}


def validate_result_row(row: Dict[str, Any]) -> None:
    """Validate that a row complies with schema and allowed enum values."""
    assert row.get("source") in VALID_SOURCES, f"Invalid source: {row.get('source')}"
    assert row.get("status") in VALID_STATUSES, f"Invalid status: {row.get('status')}"
    assert row.get("answer_grade") in VALID_ANSWER_GRADES, f"Invalid grade: {row.get('answer_grade')}"
    for k in CSV_FIELDNAMES:
        assert k in row, f"Missing column: {k}"


def append_to_results_csv(records: List[Dict[str, Any]], filepath: str = "results.csv"):
    """Append validated records to results.csv creating header if file is empty."""
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    validated = []
    for r in records:
        clean_row = {k: r.get(k, "") for k in CSV_FIELDNAMES}
        validate_result_row(clean_row)
        validated.append(clean_row)

    with open(filepath, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for r in validated:
            writer.writerow(r)

    print(f"📁 Appended {len(validated)} validated records to {filepath}")


def verify_consolidated_csv(filepath: str = "results.csv") -> Dict[str, Any]:
    """Audit results.csv and return breakdown of records by scenario and grade."""
    if not os.path.exists(filepath):
        return {"total_records": 0, "status": "FILE_NOT_FOUND"}

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDNAMES, f"Fieldnames mismatch: {reader.fieldnames}"
        rows = list(reader)

    scenarios = {}
    sources = {}
    statuses = {}
    grades = {}

    for r in rows:
        validate_result_row(r)
        scenarios[r["scenario"]] = scenarios.get(r["scenario"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        g = r["answer_grade"] or "unassigned"
        grades[g] = grades.get(g, 0) + 1

    return {
        "total_records": len(rows),
        "scenarios": scenarios,
        "sources": sources,
        "statuses": statuses,
        "grades": grades,
    }


def consolidate_all_csvs(output_path: str = "results.csv"):
    """Merge load test, retrieval, and evaluation outputs into results.csv."""
    import datetime
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    all_rows = []

    # 1. Load test rows
    if os.path.exists("results_load.csv"):
        with open("results_load.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append({k: r.get(k, "") for k in CSV_FIELDNAMES})

    # 2. Retrieval experiment rows
    if os.path.exists("retrieval_experiments.csv"):
        with open("retrieval_experiments.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append({
                    "request_id": r.get("experiment_id", ""),
                    "scenario": "retrieval_experiment",
                    "source": r.get("source", "stand-in"),
                    "start_time": now_iso,
                    "first_token_ms": "",
                    "total_ms": r.get("total_latency_ms", ""),
                    "status": "ok",
                    "retries": "",
                    "tokens_in": r.get("tokens_in", ""),
                    "tokens_out": r.get("tokens_out", ""),
                    "retrieval_setting": r.get("retrieval_setting", ""),
                    "answer_grade": "",
                })

    # 3. Answer evaluation rows
    if os.path.exists("answer_evaluation.csv"):
        with open("answer_evaluation.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append({
                    "request_id": r.get("question_id", ""),
                    "scenario": "answer_evaluation",
                    "source": "stand-in",
                    "start_time": now_iso,
                    "first_token_ms": "",
                    "total_ms": r.get("total_latency_ms", ""),
                    "status": "ok",
                    "retries": "",
                    "tokens_in": r.get("tokens_in", ""),
                    "tokens_out": r.get("tokens_out", ""),
                    "retrieval_setting": r.get("retrieval_setting", ""),
                    "answer_grade": r.get("answer_grade", ""),
                })

    # Write output
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in all_rows:
            validate_result_row(r)
            writer.writerow(r)

    print(f"✅ Successfully consolidated {len(all_rows)} rows into {output_path}")
    audit = verify_consolidated_csv(output_path)
    print("📊 Audit Breakdown:")
    print(f"   Total Records : {audit['total_records']}")
    print(f"   Scenarios     : {audit['scenarios']}")
    print(f"   Sources       : {audit['sources']}")
    print(f"   Statuses      : {audit['statuses']}")
    print(f"   Grades        : {audit['grades']}")


if __name__ == "__main__":
    consolidate_all_csvs()
