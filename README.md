# The Relay

> AI-powered research assistant with RAG, Groq LLM integration, rate limiting, retry handling, mock LLM service and load testing.

## About

**The Relay** is a Python project designed to act as an intelligent research assistant. It leverages Retrieval-Augmented Generation (RAG) and Groq LLM integration to deliver accurate, context-aware answers — with built-in support for rate limiting, retry handling, a mock LLM service for testing, and load testing capabilities.

## Project Status

🚧 **Initial Setup** — Project structure is being established.

## Requirements

- Python 3.13+
- Git

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PRINCE-MEHTA91/The-Relay.git
cd The-Relay
```

### 2. Create and activate the virtual environment

```bash
# Create the virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

---

## Load Testing

The Relay includes an asynchronous load testing harness (`load_test.py`) built with `httpx` and `asyncio` to benchmark system capacity, concurrency queues, rate limiting, and response latencies across five distinct traffic profiles.

### Five Required Load Scenarios

1. **Steady Load (`steady`)**:
   Sends paced requests at moderate concurrency to assess nominal throughput and baseline latencies.
   ```bash
   python load_test.py --scenario steady --source stand-in
   ```

2. **Burst Load (`burst`)**:
   Dispatches a simultaneous wave of concurrent requests to test bounded queue limits, queue absorption, and rate-limiting.
   ```bash
   python load_test.py --scenario burst --source stand-in
   ```

3. **Sustained Overload (`overload`)**:
   Applies high concurrency over a continuous duration to stress sliding-window rate limiters (HTTP 429), queue timeouts, and server backpressure.
   ```bash
   python load_test.py --scenario overload --source stand-in --duration 5.0
   ```

4. **Per-Minute Limit Halved (`limit_halved`)**:
   Simulates reduced upstream rate limits (15 RPM instead of 30 RPM) to verify early client-side throttling.
   ```bash
   python load_test.py --scenario limit_halved --source stand-in
   ```

5. **Daily Budget Exhausted (`budget_exhausted`)**:
   Simulates 24-hour token/request budget exhaustion using the stand-in without consuming real Groq quota.
   ```bash
   python load_test.py --scenario budget_exhausted --source stand-in
   ```

6. **All Scenarios**:
   Runs all five scenarios sequentially:
   ```bash
   python load_test.py --scenario all --source stand-in --in-process --output results_load.csv
   ```

### Additional Configuration Options

- **Streaming / Time To First Token (TTFT)**:
  Append `--stream` to stream tokens progressively via SSE and record `first_token_ms`:
  ```bash
  python load_test.py --scenario steady --source stand-in --stream
  ```

- **In-Process Testing (no external server required)**:
  Append `--in-process` to run requests directly against the FastAPI ASGI app using `httpx.ASGITransport`:
  ```bash
  python load_test.py --scenario steady --source stand-in --in-process
  ```

- **Testing Against Real Groq LLM**:
  Set `--source real` with modest request counts to preserve daily API quotas:
  ```bash
  python load_test.py --scenario steady --source real --requests 3
  ```

---

## Retrieval Quality vs. Cost Experiments

The Relay includes an experimental benchmark harness (`experiment_retrieval.py`) evaluating trade-offs between chunk size, retrieval depth (Top-K), hybrid reranking, token costs, latency, and answer citation grounding.

### Three Distinct Configurations

1. **Setting A**: Chunk size: 250 tokens, Top-K: 6, Reranking: Disabled. (Lowest cost, concise context).
2. **Setting B**: Chunk size: 500 tokens, Top-K: 4, Reranking: Disabled. (Balanced baseline context).
3. **Setting C**: Chunk size: 500 tokens, Top-K: 8, Reranking: Enabled (Hybrid lexical + semantic). (Maximal coverage).

### How to Run Experiments

```bash
# Run using the stand-in (quota safe)
python experiment_retrieval.py --source stand-in --output retrieval_experiments.csv

# Run using the real Groq API
python experiment_retrieval.py --source real --output retrieval_experiments.csv
```

---

## Answer Quality Evaluation System

The Relay includes an automated evaluation harness (`evaluate_answers.py`) grading RAG responses against 15 ground-truth research questions verified directly from the three arXiv papers (including out-of-domain negative controls).

### Evaluation Metrics & Grading

- **Four-Tier Grades**: `correct`, `wrong`, `refused_correctly`, `refused_wrongly`.
- **Citation Verification**: Checks whether the cited paper and page number match verified evidence in the actual PDFs.
- **Groundedness Validation**: Confirms citations originate from retrieved context chunks to detect hallucinations.

### How to Run Evaluation

```bash
# Evaluate with stand-in
python evaluate_answers.py --source stand-in --output answer_evaluation.csv

# Evaluate with real Groq API
python evaluate_answers.py --source real --output answer_evaluation.csv
```

---

## Consolidating All Results (`results.csv`)

Merge all experiment logs and load tests into `results.csv` complying with the exact assignment 12-column schema:

```bash
python consolidate_results.py
```

### Schema

`request_id`, `scenario`, `source`, `start_time`, `first_token_ms`, `total_ms`, `status`, `retries`, `tokens_in`, `tokens_out`, `retrieval_setting`, `answer_grade`

- **Allowed `source` values**: `real`, `stand-in`
- **Allowed `status` values**: `ok`, `rate_limited`, `timeout`, `error`
- **Allowed `answer_grade` values**: `correct`, `wrong`, `refused_correctly`, `refused_wrongly`, or empty `""`

---

## Final Report

The complete two-page assignment compliance report is located at [`final_report.md`](file:///c:/Users/91914/The-Relay/final_report.md).

## License

This project is currently unlicensed.
