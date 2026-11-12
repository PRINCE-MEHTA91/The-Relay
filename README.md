# The Relay — Production-Grade AI Research Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passing-brightgreen.svg)]()

> An asynchronous AI research assistant prototype featuring Retrieval-Augmented Generation (RAG), bounded request queueing, sliding-window rate limiting, Server-Sent Events (SSE) progressive token streaming, Groq Cloud API integration, a 7-mode deterministic stand-in, and strict citation groundedness validation.

---

## 1. Project Overview

**The Relay** is an asynchronous Python backend prototype engineered to answer complex research questions over three foundational arXiv machine learning papers:
1. *Attention Is All You Need* (Vaswani et al., arXiv:1706.03762 v7)
2. *Moshi: a speech-text foundation model for real-time dialogue* (Défossez et al., arXiv:2410.00037 v1)
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv:2507.07955 v2)

Standard RAG architectures often suffer from cascading upstream failures, memory exhaustion caused by unbounded request queues, unhandled rate-limit bursts, and hallucinations with fabricated citations. The Relay addresses these failure modes with strict concurrency guardrails, sliding-window rate shedding, evidence groundedness verification, and automated evaluation.

---

## 2. Main Features

- **Bounded Concurrency & Request Queue**: Semaphore-governed queue (capacity 10) that absorbs short bursts and rejects excess load with HTTP 503 (`Retry-After: 5`), preventing server thread lockup.
- **Server-Sent Events (SSE) Streaming**: Progressive token emission via `POST /ask/stream` or `POST /ask` (`stream=true`) providing rapid time-to-first-token (TTFT) and seamless client disconnect cleanup.
- **Sliding-Window Rate Limiting**: In-memory rate limiter enforcing requests-per-minute ceilings per client IP and issuing HTTP 429 with actionable `Retry-After` headers.
- **Upstream Resilience & Jittered Retries**: Token budget validation pre-flight, configurable request timeouts, and exponential backoff with full jitter:
  $$\text{delay} = \text{random}(0, \min(M, B \cdot 2^{\text{attempt}}))$$
- **Dual-Chunking & Hybrid Reranking**: Supports 250-token (Setting A) and 500-token (Settings B and C) chunking with dense semantic search (`all-MiniLM-L6-v2` + `faiss.IndexFlatIP`) fused with normalized lexical token overlap.
- **Strict Citation Groundedness Validator**: Validates in-text bracketed citations `[Paper Name, p. X]` against chunk metadata and checks content words against retrieved source passages to prevent hallucination.
- **Deterministic 7-Mode Stand-in**: High-fidelity mock LLM providing 7 operational modes (`NORMAL`, `SLOW`, `FAILURE`, `RATE_LIMITED`, `TIMEOUT`, `OVERLOAD`, `STREAMING`) for 100% reproducible testing without upstream quota consumption.

---

## 3. Technology Stack

- **Web Framework**: FastAPI, Starlette, Uvicorn (ASGI)
- **Retrieval & Vector Search**: FAISS (`faiss-cpu`), Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`), PyTorch, NumPy
- **Document Processing**: PyMuPDF (`fitz`), HuggingFace Tokenizers (`transformers`)
- **LLM Integration**: Groq Cloud SDK (`groq`), `python-dotenv`
- **Networking & Testing**: HTTPX (`httpx`), Pytest, unittest

---

## 4. Architecture & Folder Structure

```text
The-Relay/
├── app/
│   ├── __init__.py
│   └── main.py                                # FastAPI app, routes, SSE streaming, queue middleware
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
├── frontend/
│   ├── src/
│   │   ├── components/                        # React UI: Dashboard, ChatView, PapersLibrary, ApiStatusView, SettingsView, Sidebar, Header
│   │   ├── services/
│   │   │   └── api.js                         # SSE streaming client & REST backend connector
│   │   ├── App.jsx                            # State orchestration & persistent history/settings
│   │   ├── main.jsx                           # React root mount
│   │   └── index.css                          # Tailwind CSS v4 styling & dark theme prose
│   ├── package.json                           # React 19, Lucide icons, React Markdown, Tailwind CSS
│   └── vite.config.js                         # Vite configuration & dev proxy to backend :8000
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
│   │   ├── results.csv                        # Load test benchmark dataset
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
│   └── architecture.md                        # System architecture and flowcharts
├── .env                                       # Local configuration (untracked)
├── .env.example                               # Environment template
├── .gitignore                                 # Git ignore rules
├── LICENSE                                    # MIT License
├── README.md                                  # This documentation
└── requirements.txt                           # Project dependencies
```

---

## 5. Installation Instructions (Windows PowerShell)

### Prerequisites
- Python 3.13+ installed
- Git installed

### Setup Steps
```powershell
# 1. Clone repository
git clone https://github.com/PRINCE-MEHTA91/The-Relay.git
cd The-Relay

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 6. Environment Variable Setup

The Relay includes an `.env.example` template:

```powershell
# Copy the template to .env
Copy-Item .env.example .env
```

Configure the following variables in `.env`:
```ini
# Groq Cloud API Credentials
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024
GROQ_MAX_INPUT_TOKENS=4096
GROQ_TIMEOUT_SECONDS=30.0
GROQ_MAX_RETRIES=3
GROQ_INITIAL_RETRY_DELAY=1.0

# Rate Limiter & Concurrency Settings
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_WINDOW_SECONDS=60.0
MAX_CONCURRENCY=3
MAX_QUEUE_SIZE=5
QUEUE_TIMEOUT_SECONDS=5.0
```

> [!IMPORTANT]
> `.env` is strictly untracked and ignored by Git. Never commit or expose your real `GROQ_API_KEY`.

---

## 7. How to Run FastAPI

Start the server using Uvicorn:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- API Root: `http://127.0.0.1:8000/`
- Health Check: `http://127.0.0.1:8000/health`
- Live Metrics: `http://127.0.0.1:8000/metrics`
- Swagger UI Documentation: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

---

## 8. Frontend Web Interface (React + Vite + Tailwind CSS)

The Relay includes a complete, modern AI research assistant web interface located in `frontend/`. Inspired by Perplexity and ChatGPT, it provides researchers with an interactive graphical workspace over the three indexed machine learning papers.

### 8.1 Key Features
- **Dashboard & Suggested Queries**: Hero search box, active configuration pills, and one-click quick-start prompts mapped to the 3 research papers.
- **Progressive SSE Streaming Chat**: Real-time token-by-token streaming via `POST /ask/stream` with pulsing cursor, markdown table/formula support, and live latency/token counters.
- **Citation Cards & Grounding Audits**: Inspect each citation with interactive badges (`VERIFIED`, `REFUSAL VERIFIED`, `UNVERIFIED`) and expandable source passage drawers showing cosine similarity and chunk page numbers.
- **Research Papers Library**: Interactive catalog of the 3 indexed papers (*Attention Is All You Need*, *Moshi*, *Dynamic Chunking*) with abstracts, key architectural concepts, and direct question prompts.
- **Live Telemetry & System Status**: Live monitoring view displaying active backend health, indexed FAISS vector counts (215 vectors), rate limit ceilings, concurrency queues, and endpoint inventory.
- **Control & Settings Panel**: Adjust Top-K retrieval (1 to 10), toggle SSE streaming, switch between deterministic Mock LLM and live Groq Cloud API, test stand-in operational modes (`NORMAL`, `SLOW`, `OVERLOAD`, `FAILURE`, etc.), and manage search history.

### 8.2 How to Run Backend and Frontend

Run the Python backend and Vite frontend concurrently in two separate terminal windows:

#### Terminal 1 — Start the Backend Server (FastAPI)
```powershell
# From project root
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The FastAPI backend will be available at `http://127.0.0.1:8000`.

#### Terminal 2 — Start the Frontend Dev Server (React + Vite)
```powershell
# Navigate to frontend and start Vite
cd frontend
npm install
npm run dev
```
Open your browser and navigate to:
```text
http://localhost:5173
```

> [!NOTE]
> `frontend/vite.config.js` includes an automatic reverse proxy forwarding `/ask`, `/ask/stream`, `/health`, and `/metrics` directly to `http://127.0.0.1:8000`. This enables full SSE streaming without requiring any CORS modifications on the Python backend.

#### Production Build
To create an optimized production build of the frontend:
```powershell
cd frontend
npm run build
npm run preview
```

---

## 9. How to Run Each Test Suite

Run individual test suites from the project root:

```powershell
# 1. Load Scenarios and CSV Schema Validation
.\venv\Scripts\python.exe tests/test_load_scenarios.py

# 2. Retrieval Settings, 4-Tier Grading, and Citation Validation
.\venv\Scripts\python.exe tests/test_retrieval_and_evaluation.py

# 3. Bounded Queue & SSE Streaming Disconnect Handling
.\venv\Scripts\python.exe tests/test_queue_and_streaming.py

# 4. Sliding-Window Rate Limiter & HTTP 429
.\venv\Scripts\python.exe tests/test_rate_limiter.py

# 5. Token Budgeting, Timeouts, and Exponential Jitter Retries
.\venv\Scripts\python.exe tests/test_token_and_retries.py

# 6. FastAPI Endpoints & Mock LLM 7 Modes
.\venv\Scripts\python.exe tests/test_tasks_11_12.py
```

Run all tests via Pytest:
```powershell
.\venv\Scripts\python.exe -m pytest tests/
```

---

## 10. How to Run Experiments

### 10.1 Load Testing (`experiments/load_tests/load_test.py`)
```powershell
# Run steady load scenario with stand-in
.\venv\Scripts\python.exe experiments/load_tests/load_test.py --scenario steady --source stand-in

# Run burst load scenario
.\venv\Scripts\python.exe experiments/load_tests/load_test.py --scenario burst --source stand-in

# Run sustained overload scenario
.\venv\Scripts\python.exe experiments/load_tests/load_test.py --scenario overload --source stand-in --duration 5.0

# Run all 5 scenarios in-process
.\venv\Scripts\python.exe experiments/load_tests/load_test.py --scenario all --source stand-in --in-process
```

### 10.2 Retrieval Experiments (`experiments/retrieval/experiment_retrieval.py`)
Compares Setting A (250t, Top-6, No Rerank), Setting B (500t, Top-4, No Rerank), and Setting C (500t, Top-8, Hybrid Rerank):
```powershell
.\venv\Scripts\python.exe experiments/retrieval/experiment_retrieval.py --source stand-in
```

### 10.3 Answer Quality Evaluation (`experiments/evaluation/evaluate_answers.py`)
Evaluates 15 research queries with four-tier correctness and citation audits:
```powershell
.\venv\Scripts\python.exe experiments/evaluation/evaluate_answers.py --source stand-in
```

### 10.4 Metric Consolidation & Report Calculation (`scripts/`)
```powershell
# Audit and calculate Table 2a, 2b, and Table 3 report statistics
.\venv\Scripts\python.exe scripts/calculate_report_metrics.py

# Consolidate raw benchmark logs into results/results.csv
.\venv\Scripts\python.exe scripts/consolidate_results.py
```

---

## 11. Where to Find Reports and Raw CSV Results

### Final Reports
- **Markdown Report**: [`reports/final_report.md`](reports/final_report.md)
- **Compiled 2-Page PDF Report**: [`reports/final_report.pdf`](reports/final_report.pdf)

### Raw Benchmark Datasets
- **Master Consolidated Dataset (N=74)**: [`results/results.csv`](results/results.csv)
- **Load Test Scenarios (N=44)**: [`results/load/results_load.csv`](results/load/results_load.csv)
- **HTTP Streaming Load Runs (N=3)**: [`results/load/results_stream.csv`](results/load/results_stream.csv)
- **Batch Retrieval Experiments (N=15)**: [`results/retrieval/retrieval_experiments.csv`](results/retrieval/retrieval_experiments.csv)
- **Streaming Retrieval Experiments (N=15)**: [`results/retrieval/retrieval_streaming_experiments.csv`](results/retrieval/retrieval_streaming_experiments.csv)
- **Ground-Truth Answer Evaluation (N=15)**: [`results/evaluation/answer_evaluation.csv`](results/evaluation/answer_evaluation.csv)

---

## 12. API Endpoints & Swagger Documentation

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :---: |
| `GET` | `/` | Welcome message and API links | `200 OK` |
| `GET` | `/health` | Health check verifying FAISS index & metadata | `200 OK` / `503 Unavailable` |
| `GET` | `/metrics` | Active concurrency and rate limiter statistics | `200 OK` |
| `POST`| `/ask` | Synchronous research Q&A with citation validation | `200 OK` / `429 Rate Limited` / `503 Busy` |
| `POST`| `/ask/stream` | Server-Sent Events (SSE) progressive token stream | `200 OK` (text/event-stream) |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI | `200 OK` |
| `GET` | `/redoc` | OpenAPI ReDoc UI | `200 OK` |

### Sample Request (`POST /ask`)
```json
{
  "question": "What is the Transformer architecture and how does self-attention work?",
  "top_k": 5,
  "use_mock": true,
  "mock_mode": "NORMAL",
  "stream": false
}
```

### Sample Response (`200 OK`)
```json
{
  "question": "What is the Transformer architecture and how does self-attention work?",
  "answer": "The Transformer uses an encoder-decoder architecture with stacked self-attention and feed-forward sublayers [Attention Is All You Need, p. 3].",
  "is_valid": true,
  "validation": {
    "is_valid": true,
    "citation_errors": [],
    "unsupported_claims": [],
    "validated_citations": [
      {
        "raw_citation": "[Attention Is All You Need, p. 3]",
        "paper_name": "Attention Is All You Need",
        "page_number": 3,
        "chunk_id": 3,
        "verified": true,
        "similarity_score": 0.4856
      }
    ],
    "validation_status": "VERIFIED"
  },
  "citations": [ ... ],
  "retrieved_chunks_count": 5,
  "input_tokens": 1997,
  "output_tokens": 20,
  "retries_attempted": 0,
  "latency_ms": 128.5,
  "status": "verified"
}
```

---

## 13. License

This project is licensed under the [MIT License](LICENSE).
