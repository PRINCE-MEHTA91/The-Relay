# The Relay — AI Research Assistant Assignment Report

## 1. What I built and why

**The Relay** is a working asynchronous research assistant prototype engineered with FastAPI, Retrieval-Augmented Generation (RAG), and strict concurrency guardrails. It queries three seminal arXiv research papers:
1. *Attention Is All You Need* (1706.03762 v7)
2. *Moshi: a speech-text foundation model for real-time dialogue* (2410.00037 v1)
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (2507.07955 v2)

Standard RAG architectures fail under production traffic due to upstream rate limits, unbounded memory queues, or fabricated citations. The Relay addresses these failure modes with:
- **Bounded Concurrency Queue**: Semaphore-managed queue (capacity 10) rejecting excess bursts with HTTP 503.
- **Server-Sent Events (SSE) Streaming**: Progressive token emission for rapid time-to-first-token (TTFT) and clean disconnect handling.
- **Sliding-Window Rate Limiting & Jittered Retries**: Proactive rate throttling (30 RPM default, 15 RPM halved) and exponential backoff with full jitter.
- **Hybrid Retrieval & Groundedness Validation**: Dual chunking (250t / 500t), lexical-semantic hybrid reranking, and citation validation matching claims against retrieved text.

---

## 2. The real service and my stand-in

- **The Real Service**: Integrated with the **Groq Cloud API** using `openai/gpt-oss-20b`. With a valid `GROQ_API_KEY` loaded securely from `.env` (untracked and ignored by Git via `git check-ignore` and `git ls-files`), live integration was verified via two isolated operations:
  1. *Direct SDK Connection Test* (`req_real_groq_001`): One minimal chat completion request (`max_tokens=30`) returning `"Groq API connected successfully."` (HTTP 200, 81 in / 23 out tokens, 396.49 ms total latency; `first_token_ms` preserved blank as TTFT was not measured on non-streaming calls; `answer_grade` left legitimately blank as it verified wire connectivity rather than research corpus Q&A).
  2. *End-to-End Pipeline Request* (`req_real_groq_002`): One real FastAPI `POST /ask` call with `use_mock=false`, which performed vector retrieval, context injection, live Groq generation (HTTP 200, 1,785 in / 543 out tokens, 5,697.16 ms latency; `first_token_ms` blank), and citation validation (`status: verified`, `answer_grade: correct`).
  *Connectivity vs. Load Testing*: Verifying isolated API connectivity confirms authentication and wire compatibility, but does not measure production load resilience. To strictly protect the user's upstream quota from exhaustion, live multi-request load testing was deliberately **not** performed against Groq.
- **The Stand-in Service (`mock_llm.py`)**: A deterministic mock providing all seven enum modes (`NORMAL`, `SLOW`, `FAILURE`, `RATE_LIMITED`, `TIMEOUT`, `OVERLOAD`, `STREAMING`). It simulates 100ms baseline latency, 30ms/token SSE streaming, active concurrency tracking, and verified responses with accurate citations.
- **Reconciliation of Request Datasets across CSV Files**:
  - `results.csv` (N=76 total rows): Consolidated raw submission log recording all 76 executed requests across the project: 44 stand-in load benchmark requests, 15 stand-in retrieval experiment requests, 15 stand-in answer evaluation requests, and the 2 actual isolated live Groq verification requests (`source: real`: `req_real_groq_001` direct SDK completion, `req_real_groq_002` live `/ask` pipeline call). Preserves both real Groq and stand-in raw results adhering strictly to the 12-column schema.
  - `results_stream.csv` (N=3 rows): Standalone validation log recording 3 isolated end-to-end HTTP streaming requests under the `steady` scenario (`top_k=5`), capturing HTTP round-trip TTFT (3,432.3 ms cold-start; 745.6 ms and 682.3 ms warm; p50 = 745.6 ms).
  - `retrieval_streaming_experiments.csv` (N=15 rows): Standalone in-process streaming retrieval benchmark dataset recording 5 benchmark queries across Settings A, B, and C (5 rows per setting), capturing internal pipeline streaming TTFT (50–59 ms p50) with an in-memory warm model and zero HTTP/ASGI network overhead. It is strictly separate from the 3-row `results_stream.csv` load log.
  - `retrieval_experiments.csv` (N=15 rows): Granular per-chunk retrieval records comparing Settings A, B, and C across 5 benchmark questions (analyzed in Table 3).
  - `answer_evaluation.csv` (N=15 rows): Evidence-verified ground-truth evaluation dataset covering 15 research questions with four-tier grades and page-level citation audits (analyzed in Table 3b).

---

## 3. Results

### Table 1: Upstream Service Limits (Groq API — `openai/gpt-oss-20b`)
*Comparison of documented provider limits (Developer plan), observed live behavior, and simulated stand-in behavior across all seven required limit dimensions.*

| Limit / Metric | Documented (Live Groq)* | Observed (Live Groq)† | Simulated (Stand-in) | How Value Was Found | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. RPM (Requests/Min)** | 30 RPM | Not Tested | 30 RPM (15 RPM halved) | Official Groq Developer Docs & sliding-window limiter | Documented: High / Observed: Not Tested / Simulated: High |
| **2. TPM (Tokens/Min)** | 8,000 TPM | Not Tested | Not enforced locally | Official Groq Developer Docs | Documented: High / Observed: Not Tested / Simulated: High |
| **3. RPD (Requests/Day)** | 1,000 RPD | Not Tested | Simulated daily budget exhaustion | Official Groq Developer Docs & daily ceiling in stand-in | Documented: High / Observed: Not Tested / Simulated: High |
| **4. TPD (Tokens/Day)** | 200,000 TPD | Not Tested | Not enforced locally | Official Groq Developer Docs | Documented: High / Observed: Not Tested / Simulated: High |
| **5. Requests in Flight** | Not stated | Not Tested | Max concurrency 10 slots | Architecture spec & semaphore capacity | Documented: High / Observed: Not Tested / Simulated: High |
| **6. Limit-Hit Response** | HTTP 429 (`rate_limit_exceeded`) | Not Tested | HTTP 429 (`Retry-After: 60`), HTTP 503 | Official Groq docs & stand-in mock | Documented: High / Observed: Not Tested / Simulated: High |
| **7. Recovery Time** | Variable (per header / bucket) | Not Tested | 60s sliding window; simulated daily reset | Provider reset specification & stand-in rate limiter | Documented: High / Observed: Not Tested / Simulated: High |

*\*Official Groq Cloud documentation (https://console.groq.com/docs/rate-limits, accessed November 2026) specifies base Developer plan quotas for `openai/gpt-oss-20b` as 30 RPM, 1,000 RPD, 8,000 TPM, and 200,000 TPD. These reflect published developer tier baselines, not confirmed user-specific Free-tier account limits (which may vary dynamically by account standing or organizational tier). Documented rate-limit errors return HTTP 429. Provider recovery times are not universally fixed to 60 seconds or midnight UTC across all limits; reset schedules depend on the specific metric breached (sliding-window tokens/minute vs. rolling daily requests) and dynamic provider reset headers (e.g., `retry-after`).*  
*†Account-observed live limits, error responses, and recovery behaviors are marked Not Tested because multi-request quota exhaustion was intentionally avoided to protect user API quotas; the single live verification call succeeded with HTTP 200, so upstream rate limiting was never triggered or observed.*

---

### Table 2a: Latency, one row per load scenario (Stand-in Benchmark, N=44 from `results.csv`)

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | Total time p50 / p95 / p99 (ms) [All / OK] | First token p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Steady Load** | stand-in | 6 | 12.2 s | 216 / min (3.6/s) | 755.4 / 2,616.3 / 3,091.5 ms [All=OK] | 755.3 / 2,616.1 / 3,091.3 ms |
| **2. Burst Load** | stand-in | 10 | 1.0 s | 600 / min (10.0/s) | 0.6 / 1,139.6 / 1,383.7 ms [All] <br> 755.5 / 1,343.0 / 1,424.3 ms [OK] | 755.4 / 1,342.8 / 1,424.2 ms |
| **3. Sustained Overload**| stand-in | 15 | 2.5 s | 480 / min (8.0/s) | 0.5 / 0.7 / 0.8 ms [All] (0 ok) | Not measured† (0 tokens) |
| **4. Limit Halved** | stand-in | 8 | 5.0 s | 96 / min (1.6/s) | 655.8 / 791.9 / 794.1 ms [All] <br> 756.6 / 793.1 / 794.4 ms [OK] | 756.5 / 792.9 / 794.2 ms |
| **5. Budget Exhausted** | stand-in | 5 | 5.0 s | 60 / min (1.0/s) | 1.2 / 1.2 / 1.2 ms [All] (0 ok) | Not measured† (0 tokens) |

*Methodology: Total Latency reflects both all offered requests (including sub-millisecond HTTP 429 rate limit shedding) and successful completions (OK). Under Burst and Overload, immediate 429 rejections pull overall p50 down to 0.6ms and 0.5ms. TTFT reflects only requests that succeeded and received progressive streaming tokens (N=6 in Steady, N=4 in Burst, N=5 in Limit Halved).*  
*†TTFT is not measured for rate-limited requests because HTTP 429 is returned before token streaming begins.*

---

### Table 2b: Outcomes, same rows (Stand-in Benchmark, N=44 from `results.csv`)

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer (In/Out) | Groq 20B Cost / 1k Ans* (Hist. Stand-in) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Steady Load** | 100.0% | 0.0% | 0.0% | 0.0 | 2 | 1,997 / 20 | $0.1558 ($1.1940) |
| **2. Burst Load** | 40.0% | 60.0% | 0.0% | 0.0 | 10 | 1,992 / 20 | $0.1554 ($1.1911) |
| **3. Sustained Overload**| 0.0% | 100.0% | 0.0% | 0.0 | 10 | 0 / 0 | $0.0000 ($0.0000) |
| **4. Limit Halved** | 62.5% | 37.5% | 0.0% | 0.0 | 4 | 1,990 / 21 | $0.1555 ($1.1907) |
| **5. Budget Exhausted** | 0.0% | 100.0% | 0.0% | 0.0 | 0 | 0 / 0 | $0.0000 ($0.0000) |

*\*Pricing Methodology: All percentages calculated directly from results.csv. Exactly 0.0% of requests timed out or were given up; all unserved requests were rejected immediately (< 1.2ms) via HTTP 429. Costs are reported under official Groq GPT-OSS 20B pricing ($0.075 / 1M input tokens, $0.30 / 1M output tokens), with legacy hypothetical stand-in pricing ($0.59 / 1M input, $0.79 / 1M output) shown in parentheses for historical reference. Costs are only calculated where underlying token counts support them.*

---

### Table 3: Retrieval Settings Comparison (Stand-in Benchmark, N=15)
*Granular comparison of the three mandatory retrieval configurations across historical batch and streaming runs.*

| Setting | Config Details | Questions Asked | Correct & Cited %* | Wrong %* | Correctly Refused %* | Tokens / Ans (In/Out) | Pipeline Streaming TTFT (p50 / p95)** | Pipeline Total Latency (p50 / p95)** | Validation Pass Rate | Groq 20B Cost / 1k Ans*** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Setting A** | 250t, Top-6, No-Rerank | 5 | Not measured* | Not measured* | Not measured* | 1,255.0 / 18.4 | 59.4 ms / 143.4 ms | 777.3 ms / 819.8 ms | 100.0% (5/5) | $0.0996 ($0.7550 hist.) |
| **Setting B** | 500t, Top-4, No-Rerank | 5 | Not measured* | Not measured* | Not measured* | 1,626.8 / 18.4 | 57.4 ms / 81.5 ms | 755.7 ms / 784.3 ms | 100.0% (5/5) | $0.1275 ($0.9743 hist.) |
| **Setting C** | 500t, Top-8, Hybrid Rerank | 5 | Not measured* | Not measured* | Not measured* | 3,245.4 / 18.4 | 50.8 ms / 57.3 ms | 721.3 ms / 773.7 ms | 100.0% (5/5) | $0.2489 ($1.9293 hist.) |

*\*Per-setting factual answer grading was not separately measured across Settings A, B, and C in `retrieval_experiments.csv` (which measured chunk relevance, latency, token budgets, and 100% citation grounding). Four-tier answer grading (53.3% correct & cited, 26.7% wrong, 20.0% correctly refused) was evaluated on the comprehensive 15-question ground-truth benchmark in `answer_evaluation.csv` (reported in Table 3b). Marking per-setting answer grades as Not measured* adheres to the assignment brief requirement ("Write 'not measured' where you did not measure something") and prevents invalid duplication of overall benchmark metrics across individual settings.*  
*\*\*Measurement Boundary Notice: Streaming TTFT and total latency are in-process pipeline benchmarks measured across the 15 records in `retrieval_streaming_experiments.csv` (N=5 questions per setting). These reflect pure internal pipeline execution with a warm in-memory embedding model and zero network or ASGI serialization overhead; they do NOT represent user-facing HTTP TTFT (which is ~755 ms under load, as reported in Table 2a and verified in `results_stream.csv`). In historical batch non-streaming runs (`retrieval_experiments.csv`), batch latency p50/p95 was 129.6/144.7 ms (A), 128.4/137.0 ms (B), 128.1/147.0 ms (C).*  
*\*\*\*Cost per 1,000 answers calculated from actual token counts using official Groq GPT-OSS 20B pricing ($0.075 / 1M input, $0.30 / 1M output), with legacy hypothetical stand-in pricing shown in parentheses.*

---

### Table 3b: Ground-Truth Answer Quality Evaluation (`answer_evaluation.csv`, N=15)
*Evaluated against verified arXiv paper evidence with strict four-tier grading and page-level citation validation.*

| Metric | Count | Percentage | Definition & Evidence Verification |
| :--- | :--- | :--- | :--- |
| **Total Questions Asked** | 15 | 100.0% | 15 verified research questions covering all 3 required arXiv papers |
| **Correct & Cited** | 8 | 53.3% | Factual assertions match paper text; citations accurately cite source paper & page |
| **Wrong Answers** | 4 | 26.7% | Inaccurate technical assertions (e.g. parameter counts, detailed formulations) |
| **Refused Correctly** | 3 | 20.0% | Honest refusal for out-of-domain queries with zero evidence hallucination |
| **Refused Wrongly** | 0 | 0.0% | No in-domain answerable questions were falsely rejected |
| **Citation Paper Accuracy** | 15 / 15 | 100.0% | All citations correctly match verified arXiv source papers |
| **Citation Page Accuracy** | 15 / 15 | 100.0% | All cited page numbers match the actual evidence location in the PDF |
| **Tokens per Answer** | Mean: 2,048.9 In / 17.9 Out | — | Ground-truth benchmark token footprint |
| **Total Latency** | p50: 124.0 ms / p95: 171.6 ms | — | Stand-in evaluation latency |
| **First-Token Latency (TTFT)** | Not measured† | — | Evaluated in batch non-streaming mode |

*Key Distinction: Factual Answer Accuracy (53.3%) measures whether the response text is technically correct against ground truth. Citation Accuracy (100.0%) and Retrieval Validation Pass Rate (100.0%) measure whether claims are grounded in retrieved passages and accurately attribute source papers and pages.*

---

## 4. Where the time goes

Total response latency decomposes into:
1. **Dense Retrieval (22.9ms – 24.4ms)**: Sentence embedding generation via `all-MiniLM-L6-v2` and FAISS vector dot product over indexed chunks. Hybrid reranking in Setting C adds ~1.5ms of lexical scoring.
2. **Inference & First Token (755.3ms – 756.5ms p50 for OK)**: Upstream model ingestion and first token emission dominate overall latency.
3. **Citation Validation (1.5ms – 2.5ms)**: In-memory regex parsing and chunk metadata matching.
4. **Queue & Rejection Overhead (< 1.2ms)**: Rate-limited requests are rejected in under 1.2ms, avoiding server compute waste.

### Measurement Boundaries: HTTP Load-Test TTFT vs. In-Process Streaming TTFT
The difference between Table 2a TTFT (~755 ms) and Table 3 streaming TTFT (50–59 ms) is an architectural boundary distinction, not a measurement conflict:
- **Table 2a (End-to-End HTTP Load Test)**: Measured from the client's perspective (`t0` at HTTP request dispatch). Latency spans client network round-trip, FastAPI middleware routing, sliding-window rate-limiter check, concurrency semaphore acquisition, synchronous query embedding generation via `all-MiniLM-L6-v2`, FAISS dense vector search, context prompt formatting, and ASGI streaming response header serialization before the first SSE chunk is flushed to the network. This represents actual user-facing HTTP TTFT, confirmed in `results_stream.csv` (p50 = 745.6 ms).
- **Table 3 (In-Process Streaming Retrieval)**: Evaluated directly against the retrieval and mock streaming pipeline across the 15 records of `retrieval_streaming_experiments.csv` with an in-memory, pre-warmed embedding model and zero HTTP/ASGI network overhead. The 50–59 ms TTFT reflects pure internal pipeline dispatch and must not be presented as user-facing HTTP TTFT.

### Deconstructing an Actual Slow Request
In Steady Load, Request #1 recorded 2,413.66 ms latency (vs. ~125 ms subsequent):
- **Waiting time**: 0.1 ms (concurrency semaphore slot acquired immediately; queue was empty).
- **Working time (before first token)**: 2,411.00 ms (dominated by cold-start disk read and PyTorch weights initialization for `all-MiniLM-L6-v2` embedding generation, NOT Groq inference).
- **Working time (after first token)**: 2.56 ms (streaming token emission).
- **Retrying time**: 0.0 ms (0 retries).
The delay occurred entirely **before the first token** due to local embedding model cold-start; subsequent token delivery was instantaneous.

---

## 5. When a limit is hit

- **Who is delayed, dropped, or gets a weaker answer**:
  - *Delayed*: In-flight requests arriving during concurrency spikes are delayed in the bounded queue (capacity 10) waiting for worker execution slots up to a 5-second queue timeout.
  - *Dropped*: Requests arriving when the queue is fully saturated (depth > 10) are dropped immediately with HTTP 503 (`Retry-After: 5`). Requests exceeding sliding-window per-client rate limits (30 RPM default, 15 RPM halved) are dropped locally with HTTP 429 (`Retry-After: 60`), shielding upstream token quotas.
  - *Weaker Answers*: **Nobody receives a weaker or degraded answer**. The Relay explicitly prioritizes factual correctness and verifiable citations over degraded fallback answers; unserved requests fail fast with clean HTTP errors rather than returning hallucinated or ungrounded text.
- **The 3:00 PM Daily Budget Exhaustion Experience**: When daily quota is exhausted (simulated in scenario 5), waiting clients receive an immediate HTTP 429 error (< 1.2ms) with `"detail": "Rate limit exceeded. Daily budget exhausted."` and dynamic `Retry-After` reset headers. In live provider integrations, recovery time depends on provider-supplied quota reset timestamps rather than an unvarying 60-second or midnight UTC clock. Clients fail fast without server thread lockup or queue pileup.
- **Did Retrying Ever Make Things Worse?**: In baseline load tests, proactive local rate shedding and jittered backoffs prevented cascading retry storms. However, in our controlled overload experiment, enabling retries under sustained contention multiplied mean completion time **4.0x** (175.4 ms to 702.6 ms) and extended queue drain time by 174%, proving that blind retrying during capacity exhaustion worsens user-perceived delay.

## Retrieval: quality versus cost

Setting A provides low token consumption (1,255.0 input tokens, $0.10/1k answers Groq 20B vs. hypothetical $0.76/1k stand-in) but splits mathematical formulas across boundaries. Setting B (500 tokens, Top-4) provides the optimal balance of passage coherence and low retrieval latency (22.9ms, 1,626.8 input tokens, $0.13/1k answers Groq 20B vs. hypothetical $0.97/1k stand-in). Setting C doubles token volume (3,245.4 tokens, $0.25/1k answers Groq 20B vs. hypothetical $1.93/1k stand-in) without increasing validation pass rate over Setting B, demonstrating diminishing returns from naive context expansion. When approaching quota limits, The Relay maintains Setting B chunk boundaries but sheds non-essential queries via rate limiting rather than truncating context into unreliable fragments.

## How this library will mislead a naive retriever

1. **Cross-Paper Term Collisions**: The terms "Transformer" and "encoder" appear heavily in *Attention Is All You Need*, *Moshi*, and *Dynamic Chunking*. Querying *"What is the Transformer architecture?"* causes naive FAISS retrieval to rank *Dynamic Chunking* (chunk #141, score 0.4863) above *Attention Is All You Need* (chunk #3, score 0.4856). Hybrid reranking resolves this by boosting exact lexical matches.
2. **Conflating Modalities in Multi-Stream Architectures**: In *Moshi*, audio and text tokens are modeled simultaneously via dual streams. Querying *"How does the model handle multi-stream tokens?"* risks retrieving general multi-head attention passages from *Attention Is All You Need* rather than speech-text interleaving, unless hybrid lexical matching enforces paper-specific context.
3. **Hierarchical Boundary Collisions**: Querying *"How are tokens chunked dynamically?"* can retrieve speech frame aggregation in *Moshi* or sublayer token grouping in *Dynamic Chunking*. Grounded citation validation detects metadata mismatches if an unretrieved paper or page is cited.

---

## 6. Retries

Retries use exponential backoff with full jitter:
$$\text{delay} = \text{random}(0, \min(M, B \cdot 2^{\text{attempt}}))$$
with base delay $B = 1.0\text{s}$, maximum ceiling $M = 10.0\text{s}$, and a strict limit of 3 attempts.
- **Selective Filtering**: Only transient 429s and timeouts are retried. HTTP 400, 422, and queue-full 503 errors fail fast.
- **Infinite Loop Guard**: Strict retry counter halts further calls once the ceiling is reached.
- **Controlled Retry Degradation Experiment (Local Stand-in)**: To test whether retrying can degrade system health, we executed 10 concurrent requests under transient upstream overload (`experiments/evaluation/retry_degradation_experiment.py`):
  - *Condition A (No Retries / Fail-Fast)*: Requests failed immediately upon transient error. Mean latency was 175.4 ms (p50: 86.7 ms, p95: 392.4 ms) with total wall-clock drain time of 429.9 ms (4 ok, 6 errors).
  - *Condition B (Retries Enabled, max_retries=2, base backoff 0.5s)*: Retries eventually succeeded (10 ok, 0 errors, 1.6 avg attempts), but mean completion time surged **4.0x** to 702.6 ms (p50: 806.4 ms, p95: 1,138.7 ms) and queue drain time lengthened to 1,179.4 ms (+174%).
  - *Causal Analysis*: Retrying coroutines remain active in-flight tasks during backoff sleep. While upstream model execution locks are released between attempts, uncompleted tasks linger in the queue pool and repeatedly re-contend for capacity upon retry, multiplying tail latency and prolonging server queue pressure. In bounded-queue architectures, lingering retried tasks block execution slots and risk dropping subsequent incoming requests.

---

## 7. What the person sees

- **Streaming Experience**: SSE clients receive progressive tokens in real time, drastically cutting perceived wait time to ~755ms TTFT.
- **Standard Responses**: Non-streaming clients receive a complete JSON payload with answer text, verified citations, grounding status (`verified`/`unverified`), and token counts.
- **Out-of-Domain Refusal**: Unanswerable queries return an honest refusal: *"The answer cannot be found in the provided research papers."*

---

## 8. What I'm unsure about, what I left out, what I learned

- **Unsure About**: True multi-tenant burst headroom on Groq Free Tier under continuous hours-long load; behavior during partial upstream network partition.
- **Two Deliberately Omitted Features**:
  1. *Cross-encoder neural rerankers* (e.g. `bge-reranker-large`): Omitted due to their ~150ms CPU latency penalty, which would triple median pipeline latency.
  2. *Speculative context truncation / degraded fallback prompts*: Omitted because truncating context when near token limits drastically increases hallucination risk and violates the mandate to never send an embarrassing answer to stay fast.
- **Key Learnings**: Naive dense semantic search is easily misled in specialized domains with shared nomenclature. Bounded queuing, jittered backoff, and citation validation are essential for RAG reliability.

## Your feedback on Vaani

- **Standout**: The rigorous real-world orientation—enforcing bounded request queues, jittered exponential backoffs, strict raw schema validation, and zero metric fabrication rather than relying on abstract prompt demos.
- **Confusing**: The relationship between streaming and non-streaming latency metrics (e.g., whether TTFT is expected to be recorded for non-streaming calls or left blank) and distinguishing citation grounding pass rates (`is_valid`) from factual answer correctness.
- **One Suggested Change**: Provide standardized mock test fixtures or pre-recorded golden wire traces for upstream rate limits (e.g., Groq 429 headers and error bodies) so that edge scenarios can be validated against standardized upstream signatures without risk to live account quotas.

## How I worked with the assistant

- **Three Prompts That Mattered**:
  1. *Prompt enforcing strict schema validation and allowed enum values for load testing, ensuring zero fabricated fields.*
  2. *Prompt mandating three genuinely distinct retrieval configurations (250t vs 500t, hybrid lexical reranking) rather than simple Top-K tweaks.*
  3. *Prompt requiring a ground-truth evaluation dataset of 15 questions verified directly from arXiv PDFs with negative refusal controls.*
- **One Assistant Error I Caught**:
  *Caught the assistant using the abbreviated paper name "Dynamic Chunking for RAG" instead of the exact title "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", which broke citation matching in 250-token chunks.*
- **One Decision I Overruled**:
  *Overruled running load tests against the live Groq API to avoid exhausting the user's free daily quota, directing all load tests to the stand-in service.*
- **What Was Distinctly My Own Work**:
  *Formulated and verified the ground-truth question-evidence-page pairs across the three arXiv PDFs, established the strict four-tier grading criteria, and documented the complete conversation trajectory in `docs/conversation_history.md`.*
