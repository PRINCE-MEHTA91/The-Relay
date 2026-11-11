import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def analyze_load_results():
    filepath = find_file("results_load.csv", "load")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    with open(filepath, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)

    print("=" * 66)
    print(f"TABLE 2a & 2b METRICS BY SCENARIO (FROM {filepath.name})")
    print("=" * 66)

    for scn, scn_rows in by_scenario.items():
        total_reqs = len(scn_rows)
        all_latencies = [float(r["total_ms"]) for r in scn_rows if r["total_ms"] != ""]
        ok_latencies = [float(r["total_ms"]) for r in scn_rows if r["status"] == "ok" and r["total_ms"] != ""]
        ttfts = [float(r["first_token_ms"]) for r in scn_rows if r["first_token_ms"] != ""]
        tokens_in = [float(r["tokens_in"]) for r in scn_rows if r["tokens_in"] != ""]
        tokens_out = [float(r["tokens_out"]) for r in scn_rows if r["tokens_out"] != ""]

        status_counts = Counter(r["status"] for r in scn_rows)

        p50_all = np.percentile(all_latencies, 50) if all_latencies else 0
        p95_all = np.percentile(all_latencies, 95) if all_latencies else 0
        p99_all = np.percentile(all_latencies, 99) if all_latencies else 0

        p50_ok = np.percentile(ok_latencies, 50) if ok_latencies else 0
        p95_ok = np.percentile(ok_latencies, 95) if ok_latencies else 0
        p99_ok = np.percentile(ok_latencies, 99) if ok_latencies else 0

        p50_ttft = np.percentile(ttfts, 50) if ttfts else 0
        p95_ttft = np.percentile(ttfts, 95) if ttfts else 0
        p99_ttft = np.percentile(ttfts, 99) if ttfts else 0

        ok_pct = (status_counts["ok"] / total_reqs) * 100
        rl_pct = (status_counts["rate_limited"] / total_reqs) * 100
        to_pct = (status_counts["timeout"] / total_reqs) * 100
        err_pct = (status_counts["error"] / total_reqs) * 100

        avg_tok_in = np.mean(tokens_in) if tokens_in else 0
        avg_tok_out = np.mean(tokens_out) if tokens_out else 0
        avg_tok_total = avg_tok_in + avg_tok_out

        cost_groq_20b = ((avg_tok_in * 0.075 + avg_tok_out * 0.30) / 1000.0) if avg_tok_total > 0 else 0.0
        cost_historical = ((avg_tok_in * 0.59 + avg_tok_out * 0.79) / 1000.0) if avg_tok_total > 0 else 0.0

        print(f"\nScenario: {scn} (N={total_reqs})")
        print(f"  All Latency (ms):  p50={p50_all:.1f}, p95={p95_all:.1f}, p99={p99_all:.1f}")
        print(f"  OK Latency (ms):   p50={p50_ok:.1f}, p95={p95_ok:.1f}, p99={p99_ok:.1f}")
        print(f"  TTFT (ms, N={len(ttfts)}):     p50={p50_ttft:.1f}, p95={p95_ttft:.1f}, p99={p99_ttft:.1f}")
        print(f"  Outcomes:          ok={ok_pct:.1f}%, rate_limited={rl_pct:.1f}%, timeout={to_pct:.1f}%, error={err_pct:.1f}%")
        print(f"  Tokens/Ans:        in={avg_tok_in:.0f}, out={avg_tok_out:.0f}, total={avg_tok_total:.0f}")
        print(f"  Cost/1k (Groq 20B): ${cost_groq_20b:.4f} (Historical Stand-in: ${cost_historical:.4f})")


def analyze_retrieval_results():
    filepath = find_file("retrieval_experiments.csv", "retrieval")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    with open(filepath, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_setting = defaultdict(list)
    for r in rows:
        by_setting[r["retrieval_setting"]].append(r)

    print("\n" + "=" * 66)
    print(f"TABLE 3 METRICS BY RETRIEVAL SETTING (FROM {filepath.name})")
    print("=" * 66)

    for stg, s_rows in by_setting.items():
        n = len(s_rows)
        sims = [float(r["avg_similarity_score"]) for r in s_rows if r["avg_similarity_score"] != ""]
        tokens_in = [float(r["tokens_in"]) for r in s_rows if r["tokens_in"] != ""]
        tokens_out = [float(r["tokens_out"]) for r in s_rows if r["tokens_out"] != ""]
        ret_lats = [float(r["retrieval_latency_ms"]) for r in s_rows if r["retrieval_latency_ms"] != ""]
        tot_lats = [float(r["total_latency_ms"]) for r in s_rows if r["total_latency_ms"] != ""]
        pass_cnt = sum(1 for r in s_rows if r["is_valid"].lower() == "true")

        avg_tok_in = np.mean(tokens_in) if tokens_in else 0
        avg_tok_out = np.mean(tokens_out) if tokens_out else 0
        cost_groq_20b = ((avg_tok_in * 0.075 + avg_tok_out * 0.30) / 1000.0)
        cost_historical = ((avg_tok_in * 0.59 + avg_tok_out * 0.79) / 1000.0)

        print(f"\nSetting: {stg} (N={n})")
        print(f"  Chunk Size:    {s_rows[0]['chunk_size']}")
        print(f"  Top-K:         {s_rows[0]['top_k']}")
        print(f"  Reranking:     {s_rows[0]['reranking']}")
        print(f"  Avg Sim Score: {np.mean(sims):.3f}")
        print(f"  Tokens:        in={avg_tok_in:.1f}, out={avg_tok_out:.1f}")
        print(f"  Cost/1k Ans:   Groq 20B: ${cost_groq_20b:.4f} (Historical: ${cost_historical:.4f})")
        print(f"  Retrieval Lat: {np.mean(ret_lats):.1f} ms")
        print(f"  Total Latency: {np.mean(tot_lats):.1f} ms")
        print(f"  Pass Rate:     {pass_cnt}/{n} ({(pass_cnt/n)*100:.1f}%)")


def analyze_answer_evaluation():
    filepath = find_file("answer_evaluation.csv", "evaluation")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    with open(filepath, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 66)
    print(f"ANSWER EVALUATION AUDIT (FROM {filepath.name})")
    print("=" * 66)
    grades = Counter(r["answer_grade"] for r in rows)
    cit_paper_ok = sum(1 for r in rows if r["citation_paper_correct"].lower() == "true")
    cit_page_ok = sum(1 for r in rows if r["citation_page_correct"].lower() == "true")
    print(f"Total Questions Evaluated : {len(rows)}")
    print(f"Grades Breakdown          : {dict(grades)}")
    print(f"Correct                   : {grades.get('correct', 0)}")
    print(f"Wrong                     : {grades.get('wrong', 0)}")
    print(f"Refused Correctly         : {grades.get('refused_correctly', 0)}")
    print(f"Refused Wrongly           : {grades.get('refused_wrongly', 0)}")
    print(f"Paper Citations Correct   : {cit_paper_ok}/{len(rows)}")
    print(f"Page Citations Correct    : {cit_page_ok}/{len(rows)}")


def audit_results_csv():
    filepath = find_file("results.csv", "load")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    with open(filepath, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("\n" + "=" * 66)
    print(f"MASTER results.csv AUDIT (FROM {filepath.name})")
    print("=" * 66)
    print(f"Total rows in results.csv : {len(rows)}")
    scenarios = Counter(r["scenario"] for r in rows)
    statuses = Counter(r["status"] for r in rows)
    grades = Counter(r["answer_grade"] for r in rows)
    ttft_populated = sum(1 for r in rows if r["first_token_ms"] != "")
    print(f"Scenarios : {dict(scenarios)}")
    print(f"Statuses  : {dict(statuses)}")
    print(f"Grades    : {dict(grades)}")
    print(f"TTFT Populated Rows : {ttft_populated}/{len(rows)}")


if __name__ == "__main__":
    analyze_load_results()
    analyze_retrieval_results()
    analyze_answer_evaluation()
    audit_results_csv()
