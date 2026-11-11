import os
import sys
from pathlib import Path
import pandas as pd
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print("=" * 66)
print("COMPREHENSIVE REORGANIZATION VERIFICATION")
print("=" * 66)

# 1. Research Papers
papers = [
    "attention_is_all_you_need_v7.pdf",
    "dynamic_chunking_v2.pdf",
    "moshi_v1.pdf",
]
for p in papers:
    paper_path = PROJECT_ROOT / "papers" / p
    assert paper_path.exists(), f"Missing paper: {paper_path}"
    kb = paper_path.stat().st_size / 1024
    print(f"✅ Paper found: {p:<35} ({kb:.1f} KB)")

# 2. CSV Files
csvs = [
    ("results/results.csv", 76),
    ("results/load/results_load.csv", 44),
    ("results/load/results_stream.csv", 3),
    ("results/retrieval/retrieval_experiments.csv", 15),
    ("results/retrieval/retrieval_streaming_experiments.csv", 15),
    ("results/evaluation/answer_evaluation.csv", 15),
]
for rel_path, expected_rows in csvs:
    p = PROJECT_ROOT / rel_path
    assert p.exists(), f"Missing CSV: {p}"
    df = pd.read_csv(p)
    assert len(df) == expected_rows, f"Row count mismatch for {rel_path}: {len(df)} vs {expected_rows}"
    print(f"✅ CSV verified: {rel_path:<45} ({len(df)} rows, {len(df.columns)} cols)")

# 3. Reports
report_md = PROJECT_ROOT / "reports" / "final_report.md"
report_pdf = PROJECT_ROOT / "reports" / "final_report.pdf"
assert report_md.exists(), "Missing final_report.md"
assert report_pdf.exists(), "Missing final_report.pdf"
doc = fitz.open(str(report_pdf))
assert len(doc) == 2, f"PDF page count is {len(doc)}, expected exactly 2 pages"
print(f"✅ Report MD & PDF verified: final_report.pdf has exactly {len(doc)} pages")

# 4. Data files
assert (PROJECT_ROOT / "data" / "processed" / "chunks.json").exists()
assert (PROJECT_ROOT / "data" / "processed" / "metadata.json").exists()
assert (PROJECT_ROOT / "data" / "embeddings" / "faiss_index.bin").exists()
assert (PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy").exists()
print("✅ Data files verified: chunks, metadata, faiss_index, embeddings")

print("=" * 66)
print("ALL ASSETS VERIFIED SUCCESSFULLY!")
print("=" * 66)

