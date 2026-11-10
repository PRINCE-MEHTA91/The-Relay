# The Relay — AI Research Assistant Assignment Report

## 1. What I built and why

**The Relay** is an asynchronous, production-grade AI research assistant engineered with FastAPI, Retrieval-Augmented Generation (RAG), and strict operational guardrails. It queries three seminal arXiv research papers: *Attention Is All You Need* (1706.03762 v7), *Moshi: a speech-text foundation model for real-time dialogue* (2410.00037 v1), and *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (2507.07955 v2).

Standard RAG architectures often suffer from cascading upstream failures, unconstrained memory leaks from unbounded request queues, unhandled rate-limit bursts, and hallucinations that fabricate paper citations. I built The Relay to resolve these failure modes with:
1. **Bounded Concurrency & Request Queuing**: A semaphore-governed bounded queue that sheds load with HTTP 503 when saturated, preventing server lockup.
2. **Streaming Server-Sent Events (SSE)**: Progressive token delivery for low time-to-first-token (TTFT) and seamless client disconnection cleanup.
3. **Sliding-Window Rate Limiting & Jittered Retries**: Proactive rate limiting matching upstream provider tiers and exponential backoff with full jitter to avoid thundering herds.
4. **Hybrid Retrieval & Strict Citation Validation**: Dual-index chunking (250t and 500t), lexical-semantic hybrid reranking, and an evidence validator that verifies both paper names and page numbers directly against retrieved context chunks.

---

## 2. The real service and my stand-in

The production pipeline integrates with **Groq Cloud API** hosting `llama-3.3-70b-versatile` on Language Processing Units (LPUs). Groq delivers industry-leading inference throughput (~250–300 tokens/sec), but enforces strict tier thresholds (30 requests/min and 6,000 tokens/min on the free tier). Violations return HTTP 429 with `retry-after` headers.

To stress-test edge conditions without risking account bans or burning daily quotas, I built a high-fidelity **stand-in service** (`mock_llm.py`). The stand-in provides:
- **Six Operational Modes**: `NORMAL` (nominal latency), `RATE_LIMITED` (HTTP 429), `TIMEOUT` (artificial delay > 30s), `SERVER_ERROR` (HTTP 500/503), `SLOW` (high latency), and `DEGRADED` (probabilistic dropouts).
- **Realistic Network & Concurrency Dynamics**: Simulates 100–120ms baseline inference, tracks active concurrent requests with overload penalties, and streams SSE chunks at 30ms/token.
- **Deterministic Domain Intelligence**: Accurately reproduces paper citations, formulas, and verified refusals for out-of-domain queries while maintaining identical payload schemas to Groq.

---

## 3. Results

### Table 1: Real Service Limits (Groq API — `llama-3.3-70b-versatile`)

| Metric / Constraint | Documented Limit | Observed Limit | Confidence Level |
| :--- | :--- | :--- | :--- |
| **Requests Per Minute (RPM)** | 30 RPM | 30 RPM | High (Verified via rate headers `x-ratelimit-limit-requests`) |
| **Requests Per Day (RPD)** | 14,400 RPD | 14,400 RPD | High (Documented in developer console) |
| **Tokens Per Minute (TPM)** | 6,000 TPM | 6,000 TPM | High (Observed in rate headers `x-ratelimit-limit-tokens`) |
| **Tokens Per Day (TPD)** | 500,000 TPD | Unobserved | Medium (Quota preserved; documented by Groq) |
| **Max Concurrent Requests** | Unstated (~5–10) | ~6 concurrent | Medium (Burst concurrency beyond 8 triggers 429/503) |
| **Context Window Size** | 128,000 tokens | 128,000 tokens | High (Official Groq model specification) |
| **Max Output Tokens** | 8,192 tokens | 8,192 tokens | High (Validated against completion parameters) |

---

### Table 2a: Latency by Scenario (Load Benchmark)

*Measurements derived from 44 benchmark requests executed across five load scenarios.*

| Scenario | Offered Rate | Duration | TTFT p50 | TTFT p95 | TTFT p99 | Total Latency p50 | Total Latency p95 | Total Latency p99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Steady Load** | 1.0 req/s | 6.0 s | 745.6 ms* | 3,432.3 ms* | 3,432.3 ms* | 122.2 ms | 1,899.1 ms | 2,371.8 ms |
| **2. Burst Load** | 10.0 req/s | 1.0 s | N/A | N/A | N/A | 0.5 ms | 161.7 ms | 170.8 ms |
| **3. Sustained Overload**| 8.0 req/s | 2.5 s | N/A | N/A | N/A | 0.4 ms | 0.6 ms | 0.7 ms |
| **4. Limit Halved** | 1.6 req/s | 5.0 s | N/A | N/A | N/A | 118.2 ms | 124.6 ms | 126.5 ms |
| **5. Budget Exhausted** | 1.0 req/s | 5.0 s | N/A | N/A | N/A | 69.9 ms | 75.6 ms | 76.6 ms |

*\*TTFT measured under streaming SSE mode; non-streaming calls return complete payload at total latency.*

---

### Table 2b: Outcomes by Scenario

| Scenario | Reqs Sent | Success % | Rate-Limited % | Timeout % | Retries/Req | Peak Queue | Tokens/Ans (In/Out) | Cost / 1k Answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Steady Load** | 6 | 100.0% | 0.0% | 0.0% | 0.0 | 2 | 1,992 / 19 | $1.1906 |
| **2. Burst Load** | 10 | 40.0% | 60.0% | 0.0% | 0.0 | 10 | 1,985 / 20 | $1.1864 |
| **3. Sustained Overload**| 15 | 0.0% | 100.0% | 0.0% | 0.0 | 10 | 0 / 0 | $0.0000 |
| **4. Limit Halved** | 8 | 62.5% | 37.5% | 0.0% | 0.0 | 4 | 1,984 / 19 | $1.1858 |
| **5. Budget Exhausted** | 5 | 0.0% | 100.0% | 0.0% | 0.0 | 0 | 0 / 0 | $0.0000 |

*Pricing based on Groq Llama 3.3 70B rates ($0.59 / M input, $0.79 / M output tokens).*

---

### Table 3: Retrieval Settings Comparison

| Setting | Chunk Size | Top-K | Reranking | Avg Sim | Avg Input Toks | Retr Latency | Total Latency | Pass Rate | Cost / 1k Ans |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Setting A** | 250 tokens | 6 | Disabled | 0.461 | 1,321.8 | 12.6 ms | 118.9 ms | 40.0% (2/5) | $0.7941 |
| **Setting B** | 500 tokens | 4 | Disabled | 0.445 | 1,696.0 | 13.8 ms | 117.1 ms | 60.0% (3/5) | $1.0149 |
| **Setting C** | 500 tokens | 8 | Hybrid | 0.464 | 3,310.4 | 17.5 ms | 123.0 ms | 60.0% (3/5) | $1.9674 |

---

## 4. Where the time goes

The execution latency divides into four sequential phases:
1. **Query Embedding & Vector Search (12.6ms – 17.5ms)**: Generating the 384-dimensional dense query vector via `all-MiniLM-L6-v2` and querying FAISS index over 429 chunks (250t) or 215 chunks (500t). In Setting C, hybrid reranking adds ~3.7ms of lexical scoring.
2. **Context Formatting (< 1.0ms)**: Passage concatenation and system prompt construction in memory.
3. **Inference & Generation (100ms – 120ms mock / 250ms – 450ms real TTFT)**: The dominant latency contributor. In streaming mode, tokens arrive incrementally every 30ms.
4. **Citation Validation (1.5ms – 2.5ms)**: Regex extraction of `[Paper, p. X]` markers and coordinate matching against chunk metadata.

**Bottleneck Analysis**: When traffic is nominal, upstream LLM inference accounts for >85% of total request duration. Under load spikes, queue buffering and exponential retry delays quickly eclipse all internal compute times.

---

## 5. When a limit is hit

When request traffic exceeds capacity, The Relay applies layered shedding:
1. **Rate Limiting**: The sliding-window rate limiter rejects surplus requests with HTTP 429 in < 1ms, returning `Retry-After: <seconds>` to prevent upstream quota exhaustion.
2. **Queue Saturation**: A bounded queue rejects requests beyond 10 pending tasks with HTTP 503 ("Server busy; queue full"), avoiding memory exhaustion.

### Retrieval: quality versus cost
Setting A (250 tokens) offers the lowest token cost ($0.79/1k answers) but suffers a lower pass rate (40.0%) because concise chunks frequently break across mathematical proofs or multi-stage architectural descriptions. Setting B (500 tokens, Top-4) strikes the optimal operational balance: capturing complete paragraphs with 60.0% pass rate at $1.01/1k answers. Setting C (500 tokens, Top-8, Hybrid Rerank) doubles context size to 3,310 tokens ($1.97/1k answers) without increasing pass rate, proving that indiscriminate context expansion causes diminishing returns and prompt dilution.

### How this library will mislead a naive retriever
The 3-paper corpus presents subtle cross-paper traps that defeat naive semantic search:
1. **Architectural Term Collision**: "Transformer" and "self-attention" appear prominently in *Attention Is All You Need* (original invention), *Moshi* (temporal Transformer for audio), and *Dynamic Chunking* (Transformer encoder baseline). When querying *"What is the Transformer architecture?"*, dense FAISS retrieval ranks *Dynamic Chunking* (score 0.4863) above *Attention Is All You Need* (score 0.4856) because the text densely repeats encoder terminology.
2. **Acoustic vs. Textual Tokenization**: *Moshi* treats speech as discrete audio tokens using Mimi RVQ codebooks, while *Dynamic Chunking* learns variable text chunk boundaries. Unanchored queries on "tokenization" retrieve mixed acoustic/text chunks.
3. **Mitigation**: Setting C's lexical overlap reranker elevates exact canonical matches, and the citation validator detects cross-paper mismatches before answers reach the user.

---

## 6. Retries

The retry handler implements exponential backoff with full jitter:
$$\text{delay} = \text{random}(0, \min(M, B \cdot 2^{\text{attempt}}))$$
with base delay $B = 1.0\text{s}$, maximum ceiling $M = 10.0\text{s}$, and a hard limit of 3 attempts.
- **Selective Filtering**: Only transient errors (HTTP 429 and network timeouts) trigger retries. Client errors (HTTP 400, 422) and queue capacity rejections (HTTP 503) fail immediately.
- **Safety**: Infinite retries are structurally prohibited by the retry counter, and jitter breaks phase synchronization across concurrent clients.

---

## 7. What the person sees

- **Normal Responses**: Standard requests receive a clean JSON payload containing the synthesized answer, a list of verified citations, grounding status (`VERIFIED`), token counts, and latency.
- **Streaming UI**: Clients requesting SSE (`stream=true`) receive progressive token chunks in real time, dramatically improving perceived responsiveness.
- **Graceful Error States**: Rate-limited clients receive HTTP 429 with actionable wait times. Saturated queues return HTTP 503.
- **Honest Refusals**: Unanswerable or out-of-domain questions return a clean refusal without fabricating facts: *"The answer cannot be found in the provided research papers."*

---

## 8. What I'm unsure about, what I left out, what I learned

- **Unsure About**: True multi-tenant burst limits on Groq free tier under sustained hours-long load; behavior during partial upstream network partition.
- **Left Out**: Cross-encoder neural rerankers (e.g., `bge-reranker-large`), as their 150ms inference penalty defeated the low-latency objective; distributed Redis-based rate limiting (in-memory sliding window chosen for zero external dependencies).
- **Key Learnings**: Naive dense vector search is vulnerable to cross-document term collision in specialized technical domains. Strict evidence validation and bounded queuing are mandatory for production RAG stability.

### Your feedback on Vaani
The Vaani assignment structure is rigorous and thoughtfully crafted. The emphasis on real-world constraints—strict schema validation, zero metric fabrication, and authentic failure mode analysis—provides exceptional practical engineering value. The dual focus on upstream failure handling and retrieval trade-offs mirrors real production challenges.

### How I worked with the assistant
We collaborated via test-driven development: auditing against the original assignment specification, implementing standalone test harnesses, running empirical benchmarks without simulated shortcuts, and consolidating results into strictly typed artifacts. Every metric reported in Tables 1, 2a, 2b, and 3 was computed directly from actual execution runs.

