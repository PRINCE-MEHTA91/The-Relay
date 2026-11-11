# The Relay — System Architecture & Component Design

## 1. System Overview

**The Relay** is an asynchronous AI research assistant prototype designed for high reliability, bounded resource utilization, and evidence-grounded responses over scientific literature.

The system indexes and queries three arXiv research papers:
1. *Attention Is All You Need* (Vaswani et al., 1706.03762 v7)
2. *Moshi: a speech-text foundation model for real-time dialogue* (Défossez et al., 2410.00037 v1)
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (2507.07955 v2)

```mermaid
flowchart TD
    Client[HTTP Client / SSE Consumer] -->|POST /ask or POST /ask/stream| API[FastAPI Application (app/main.py)]
    API --> RateLimiter[Sliding-Window Rate Limiter (core/rate_limiter.py)]
    RateLimiter -->|Allow| Queue[Concurrency Semaphore Queue (Capacity 10)]
    RateLimiter -->|Exceed| R429[HTTP 429 Rate Limit Exceeded]
    Queue -->|Queue Full| R503[HTTP 503 Server Busy]
    Queue -->|Slot Acquired| Retrieval[FAISS Vector & Hybrid Search (rag/search_papers.py)]
    Retrieval --> ModelSelect{Provider Selector}
    ModelSelect -->|use_mock=False| Groq[Groq Cloud API: openai/gpt-oss-20b (core/groq_service.py)]
    ModelSelect -->|use_mock=True| Mock[Deterministic Stand-in (core/mock_llm.py)]
    Groq --> Validator[Citation & Groundedness Validator (rag/citation_validator.py)]
    Mock --> Validator
    Validator --> Response[JSON / SSE Stream with Citations & Metadata]
```

---

## 2. Directory Structure

```text
The-Relay/
├── app/
│   ├── __init__.py
│   └── main.py                                # FastAPI endpoints, queue middleware, SSE streaming
├── core/
│   ├── __init__.py                            # Core package exports
│   ├── answer_generator.py                    # RAG pipeline orchestration and context formatting
│   ├── groq_service.py                        # Groq Cloud SDK integration, token budgets, retries
│   ├── mock_llm.py                            # Stand-in deterministic mock with 7 modes
│   └── rate_limiter.py                        # Sliding-window rate limiter & concurrency semaphore
├── rag/
│   ├── __init__.py                            # RAG package exports
│   ├── chunk_papers.py                        # Multi-size token window chunking (250t / 500t)
│   ├── citation_validator.py                  # Strict paper/page regex & passage grounding audit
│   ├── extract_papers.py                      # PDF text extraction with PyMuPDF
│   ├── generate_embeddings.py                 # SentenceTransformers all-MiniLM-L6-v2 embeddings
│   └── search_papers.py                       # FAISS IndexFlatIP cosine search & hybrid reranker
├── papers/
│   ├── attention_is_all_you_need_v7.pdf       # arXiv:1706.03762 v7
│   ├── dynamic_chunking_v2.pdf                # arXiv:2507.07955 v2
│   ├── moshi_v1.pdf                           # arXiv:2410.00037 v1
│   └── download_papers.py                     # Paper download automation
├── data/
│   ├── embeddings/
│   │   ├── embeddings.npy                     # Dense sentence embeddings (384 dims)
│   │   ├── faiss_index.bin                    # FAISS IndexFlatIP (500-token chunks)
│   │   └── faiss_index_250.bin                # FAISS IndexFlatIP (250-token chunks)
│   ├── extracted_text/
│   │   ├── attention_is_all_you_need.txt
│   │   ├── dynamic_chunking.txt
│   │   └── moshi.txt
│   └── processed/
│       ├── chunk_metadata.json
│       ├── chunks.json                        # 500-token chunks with page metadata
│       ├── chunks_250.json                    # 250-token chunks for Setting A
│       └── metadata.json
├── experiments/
│   ├── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_answers.py                # 15-question ground-truth evaluation harness
│   ├── load_tests/
│   │   ├── __init__.py
│   │   └── load_test.py                       # 5-scenario load testing runner
│   └── retrieval/
│       ├── __init__.py
│       └── experiment_retrieval.py            # Retrieval settings A, B, C comparison harness
├── results/
│   ├── results.csv                            # Master consolidated 74-request benchmark log
│   ├── evaluation/
│   │   └── answer_evaluation.csv              # 15-question 4-tier grading benchmark
│   ├── load/
│   │   ├── results.csv
│   │   ├── results_load.csv                   # 44-request load benchmark dataset
│   │   └── results_stream.csv                 # 3-request HTTP streaming benchmark dataset
│   └── retrieval/
│       ├── retrieval_experiments.csv          # 15-request batch retrieval experiment dataset
│       └── retrieval_streaming_experiments.csv# 15-request streaming retrieval benchmark
├── reports/
│   ├── final_report.md                        # Final technical report (Markdown)
│   └── final_report.pdf                       # Submission-ready 2-page compiled PDF report
├── scripts/
│   ├── __init__.py
│   ├── calculate_report_metrics.py            # Quantitative report metric generator
│   ├── consolidate_results.py                 # CSV audit and consolidation utility
│   └── verify_groq_connection.py              # Safe single-request Groq wire verification
├── tests/
│   ├── __init__.py
│   ├── test_ask_live_groq.py                  # Single-request live endpoint verification
│   ├── test_load_scenarios.py                 # All 5 load scenarios & CSV schema audit
│   ├── test_queue_and_streaming.py            # Bounded queue & SSE disconnect tests
│   ├── test_rate_limiter.py                   # Sliding-window unit and integration tests
│   ├── test_retrieval_and_evaluation.py       # Retrieval settings & 4-tier grading tests
│   ├── test_tasks_11_12.py                    # FastAPI endpoints & MockLLM modes tests
│   └── test_token_and_retries.py              # Token budgets & exponential jitter tests
├── docs/
│   └── architecture.md                        # This architecture documentation
├── .env                                       # Local configuration (untracked)
├── .env.example                               # Environment template
├── .gitignore                                 # Git ignore rules
├── LICENSE                                    # MIT License
├── README.md                                  # Project overview and instructions
└── requirements.txt                           # Project dependencies
```

---

## 3. Core Subsystems

### 3.1 Concurrency & Rate Limiting (`core/rate_limiter.py`)
- **Sliding-Window Limiter**: Tracks request timestamps in memory per client IP over a 60-second sliding window. Saturated quotas return HTTP 429 with `Retry-After` headers.
- **Bounded Semaphore Queue**: Enforces a strict concurrency limit (default: 3 active slots, queue capacity: 5 or 10 slots). Overflow bursts are rejected immediately with HTTP 503 ("Server busy: request queue is full"), preventing worker pool exhaustion.

### 3.2 Upstream Integration & Stand-in (`core/groq_service.py`, `core/mock_llm.py`)
- **Live Groq Client**: Connects securely to Groq Cloud API using `openai/gpt-oss-20b` with pre-flight token budgeting (`estimate_tokens`), configurable timeouts, and exponential backoff with full jitter:
  $$\text{delay} = \text{random}(0, \min(M, B \cdot 2^{\text{attempt}}))$$
- **Mock LLM Service**: A deterministic, zero-quota stand-in providing all 7 enum modes (`NORMAL`, `SLOW`, `FAILURE`, `RATE_LIMITED`, `TIMEOUT`, `OVERLOAD`, `STREAMING`).

### 3.3 Retrieval-Augmented Generation (`rag/`)
- **Text Extraction**: PyMuPDF extracts text per page, preserving exact page boundary markers.
- **Dual Chunking**: Builds 250-token chunks with 25-token overlap (Setting A) and 500-token chunks with 50-token overlap (Settings B and C).
- **Dense Vector Search**: Encodes queries and chunks with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and performs exact cosine similarity search via `faiss.IndexFlatIP`.
- **Hybrid Reranking**: Combines dense semantic dot products with normalized lexical n-gram token overlap:
  $$\text{score}_{\text{hybrid}} = \alpha \cdot \text{score}_{\text{sem}} + (1 - \alpha) \cdot \text{score}_{\text{lex}}$$
- **Citation Validation**: Parses in-text bracketed citations `[Paper, p. X]`, audits them against retrieved chunk metadata, and matches claim phrases against passage text to prevent hallucinated citations.

---

## 4. Benchmark Dataset Reconciliation

All project benchmarks are recorded in strictly structured CSV files:
- `results/results.csv` (N=74): Consolidated master log covering 44 load-test requests, 15 retrieval experiment requests, and 15 answer evaluation requests.
- `results/load/results_load.csv` (N=44): Five load scenarios (steady, burst, sustained overload, limit halved, daily budget exhausted).
- `results/load/results_stream.csv` (N=3): Isolated end-to-end HTTP streaming requests measuring client TTFT.
- `results/retrieval/retrieval_experiments.csv` (N=15): Granular chunk-level comparison of Settings A, B, and C across 5 research queries.
- `results/retrieval/retrieval_streaming_experiments.csv` (N=15): In-process pipeline streaming benchmarks across Settings A, B, and C.
- `results/evaluation/answer_evaluation.csv` (N=15): Verified ground-truth benchmark with four-tier correctness grading and page-level citation validation.

