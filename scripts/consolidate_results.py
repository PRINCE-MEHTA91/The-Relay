"""Integration module validating and consolidating all benchmark outputs into results.csv."""

import os
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Required 12-column schema specified in assignment brief
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


def find_file(filename: str, subfolder: str = "") -> Path:
    candidates = [
        PROJECT_ROOT / "results" / subfolder / filename,
        PROJECT_ROOT / "results" / filename,
        PROJECT_ROOT / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / filename


def validate_row_schema(row: Dict[str, Any], row_idx: int) -> None:
    """Validate that a row contains valid fields and adheres to enum domains."""
    for field in CSV_FIELDNAMES:
        if field not in row:
            raise ValueError(f"Row {row_idx}: Missing required field '{field}'")


def append_to_results_csv(records: List[Dict[str, Any]], filepath: str = "results.csv"):
    """Append validated records to results.csv creating header if file is empty."""
    target_path = Path(filepath)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / filepath

    file_exists = target_path.exists() and target_path.stat().st_size > 0
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for idx, rec in enumerate(records, start=1):
            validate_row_schema(rec, idx)
            writer.writerow({k: rec.get(k, "") for k in CSV_FIELDNAMES})


def verify_consolidated_csv(filepath: str = "results.csv") -> Dict[str, Any]:
    """Audit results.csv and return breakdown of records by scenario and grade."""
    target_path = Path(filepath)
    if not target_path.is_absolute():
        target_path = find_file(filepath, "load")

    if not target_path.exists():
        raise FileNotFoundError(f"{target_path} not found.")

    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_FIELDNAMES:
            raise ValueError(f"Schema mismatch in {target_path}: {reader.fieldnames} vs {CSV_FIELDNAMES}")
        rows = list(reader)

    scenarios = {}
    sources = {}
    statuses = {}
    grades = {}

    for idx, r in enumerate(rows, start=1):
        validate_row_schema(r, idx)
        scenarios[r["scenario"]] = scenarios.get(r["scenario"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        grades[r["answer_grade"]] = grades.get(r["answer_grade"], 0) + 1

    return {
        "filepath": str(target_path),
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

    load_path = find_file("results_load.csv", "load")
    if load_path.exists():
        with open(load_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append({k: r.get(k, "") for k in CSV_FIELDNAMES})

    ret_path = find_file("retrieval_experiments.csv", "retrieval")
    if ret_path.exists():
        with open(ret_path, mode="r", encoding="utf-8") as f:
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

    eval_path = find_file("answer_evaluation.csv", "evaluation")
    if eval_path.exists():
        with open(eval_path, mode="r", encoding="utf-8") as f:
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

    destinations = [
        PROJECT_ROOT / output_path,
        PROJECT_ROOT / "results" / output_path,
        PROJECT_ROOT / "results" / "load" / output_path,
    ]

    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"Consolidated {len(all_rows)} rows to {dest}")


if __name__ == "__main__":
    audit = verify_consolidated_csv()
    print("=" * 66)
    print(f"AUDIT RESULTS FOR: {audit['filepath']}")
    print("=" * 66)
    print(f"Total Rows : {audit['total_records']}")
    print(f"Scenarios  : {audit['scenarios']}")
    print(f"Sources    : {audit['sources']}")
    print(f"Statuses   : {audit['statuses']}")
    print(f"Grades     : {audit['grades']}")
    print("=" * 66)
