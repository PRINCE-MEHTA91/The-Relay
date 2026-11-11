# The Relay — Conversation History & Prompt Log

This document records the prompt progression, tool interactions, errors caught, decisions overruled, and independent verification work conducted while collaborating with the AI coding assistant (Antigravity) on **The Relay** assignment.

---

## 1. Important Prompts & Why They Mattered

### Prompt 1: Strict Raw Schema Enforcement and Allowed Enum Values
- **The Prompt**:
  > *"Validate results.csv against the exact 12 columns in the brief: request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade. Enforce allowed enum values: source (real, stand-in), status (ok, rate_limited, timeout, error), answer_grade (correct, wrong, refused_correctly, refused_wrongly). Ensure zero fabricated metrics."*
- **Why It Mattered**:
  AI assistants tend to invent columns or populate empty fields with synthetic data. Enforcing strict schema validation ensured every row in `results.csv` represents a genuine request record with reproducible metrics, valid ISO timestamps, and exact allowed enums.

### Prompt 2: Three Genuinely Distinct Retrieval Configurations
- **The Prompt**:
  > *"Do not merely vary top_k. Implement and benchmark three genuinely distinct retrieval configurations: Setting A (250-token chunks, top-6, no reranking), Setting B (500-token chunks, top-4, no reranking), and Setting C (500-token chunks, top-8, with hybrid lexical-semantic reranking). Recompute embeddings and evaluate latency and token footprint across all three."*
- **Why It Mattered**:
  A naive approach only tests different `top_k` values on a single index. This prompt forced the creation of a distinct 250-token index (`faiss_index_250.bin`) and a lexical-semantic hybrid reranking pipeline, revealing the true trade-off between passage granularity, formula preservation, and token consumption.

### Prompt 3: Evidence-Verified Ground-Truth Evaluation Dataset
- **The Prompt**:
  > *"Create an answer evaluation dataset of 15 questions covering all three arXiv research papers plus out-of-domain negative controls. Verify every question against the actual text and page numbers in the PDFs. Evaluate factual accuracy separately from citation validation pass rate."*
- **Why It Mattered**:
  This prevented the conflation of retrieval validation pass rate (`is_valid`) with factual answer correctness. It created the 15-question benchmark in `answer_evaluation.csv` that measured 53.3% factual accuracy (8/15), 26.7% wrong (4/15), 20.0% correctly refused (3/15), alongside 100% citation grounding (15/15).

---

## 2. Assistant Errors Caught & How They Were Caught

### Error 1: Abbreviated Research Paper Title
- **What Happened**: The assistant initialized chunk metadata for the third paper with the abbreviated title `"Dynamic Chunking for RAG"`.
- **How It Was Caught**: When running `citation_validator.py`, citations referencing the full paper title failed validation because the string did not match. Inspecting the actual PDF title confirmed the title is `"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"`.
- **Resolution**: Updated `extract_papers.py`, `chunk_papers.py`, and `citation_validator.py` to use the full arXiv title.

### Error 2: Pytest Incompatibility with Async Test Functions
- **What Happened**: `test_mock_llm_modes` in `tests/test_tasks_11_12.py` was defined as an `async def` function at module level, causing Pytest to raise an error because `pytest-asyncio` was not installed.
- **How It Was Caught**: Running `pytest -v tests/` flagged `test_mock_llm_modes` as uncallable.
- **Resolution**: Wrapped the async execution into a standard synchronous wrapper `def test_mock_llm_modes(): asyncio.run(_run_mock_llm_modes())`.

---

## 3. Decisions Where the Assistant Was Overruled

### Overruling Live Groq Load Testing
- **What Happened**: The assistant proposed running multi-request concurrent load tests against the live Groq API endpoint.
- **Why Overruled**: The user was operating on a free Groq account. Running 44 concurrent benchmark requests would have quickly exhausted the daily request and token limits (1,000 RPD / 200,000 TPD), blocking further testing.
- **Resolution**: Instructed the assistant to use the live Groq API strictly for isolated single-request connectivity and wire-compatibility verification, while directing all 44 load benchmark requests to the local stand-in service.

---

## 4. Distinctly User-Owned Work

1. **Ground-Truth Question & Evidence Formulation**:
   - Formulated the 15 research questions spanning *Attention Is All You Need* (scaled dot-product, complexity, multi-head attention), *Moshi* (Helium LM, Mimi RVQ, dual audio streams), and *Dynamic Chunking* (routing module, smoothing module, monotone width scaling), plus negative out-of-domain controls (French croissants, quantum syndromes).
   - Manually mapped each question to exact page numbers in the source arXiv PDFs.

2. **Grading Criteria & Metric Distinction**:
   - Established the strict four-tier grading framework (`correct`, `wrong`, `refused_correctly`, `refused_wrongly`).
   - Enforced the critical architectural separation between citation validation pass rate (`is_valid`) and factual correctness (`answer_grade`).

3. **System Architectural Boundaries**:
   - Defined the boundary distinction between end-to-end HTTP TTFT (~755 ms under load, including network and ASGI overhead) versus pure in-process pipeline streaming TTFT (50–59 ms).

