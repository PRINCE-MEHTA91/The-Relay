"""Comprehensive assignment auditor checking code, data, tables, and metrics against the brief."""

import os
import sys
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def audit_results_csv():
    csv_path = PROJECT_ROOT / "results" / "results.csv"
    if not csv_path.exists():
        return False, ["results/results.csv does not exist"]

    expected_header = [
        "request_id", "scenario", "source", "start_time", "first_token_ms",
        "total_ms", "status", "retries", "tokens_in", "tokens_out",
        "retrieval_setting", "answer_grade"
    ]
    valid_sources = {"real", "stand-in"}
    valid_statuses = {"ok", "rate_limited", "timeout", "error"}
    valid_grades = {"", "correct", "wrong", "refused_correctly", "refused_wrongly"}

    errors = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, ["results.csv is empty"]

        if header != expected_header:
            errors.append(f"Header mismatch: expected {expected_header}, got {header}")

        rows = list(reader)

    if len(rows) != 76:
        errors.append(f"Expected 76 rows (74 stand-in + 2 real), got {len(rows)}")

    seen_ids = set()
    for idx, row in enumerate(rows, start=2):
        if len(row) != 12:
            errors.append(f"Line {idx}: expected 12 columns, got {len(row)}")
            continue

        req_id, scn, src, st_time, ft_ms, tot_ms, stat, retries, tin, tout, r_set, grade = row
        if req_id in seen_ids:
            errors.append(f"Line {idx}: duplicate request_id {req_id}")
        seen_ids.add(req_id)

        if src not in valid_sources:
            errors.append(f"Line {idx}: invalid source '{src}'")
        if stat not in valid_statuses:
            errors.append(f"Line {idx}: invalid status '{stat}'")
        if grade not in valid_grades:
            errors.append(f"Line {idx}: invalid answer_grade '{grade}'")

        try:
            datetime.fromisoformat(st_time.replace("Z", "+00:00"))
        except Exception:
            errors.append(f"Line {idx}: invalid timestamp '{st_time}'")

        if ft_ms != "":
            try:
                if float(ft_ms) < 0:
                    errors.append(f"Line {idx}: negative first_token_ms")
            except ValueError:
                errors.append(f"Line {idx}: non-float first_token_ms '{ft_ms}'")

        if tot_ms != "":
            try:
                if float(tot_ms) < 0:
                    errors.append(f"Line {idx}: negative total_ms")
            except ValueError:
                errors.append(f"Line {idx}: non-float total_ms '{tot_ms}'")

    return (len(errors) == 0), errors

def audit_pdf_page_count():
    pdf_path = PROJECT_ROOT / "reports" / "final_report.pdf"
    if not pdf_path.exists():
        return False, "reports/final_report.pdf does not exist"
    try:
        import fitz
        doc = fitz.open(pdf_path)
        count = len(doc)
        if count == 2:
            return True, f"final_report.pdf has exactly {count} pages"
        else:
            return False, f"final_report.pdf has {count} pages (must be <= 2)"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    ok_csv, csv_errs = audit_results_csv()
    print("CSV Audit:", "PASS" if ok_csv else "FAIL")
    if not ok_csv:
        for err in csv_errs[:5]:
            print("  ", err)
    ok_pdf, pdf_msg = audit_pdf_page_count()
    print("PDF Audit:", "PASS" if ok_pdf else "FAIL", "-", pdf_msg)

