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

The Relay includes an asynchronous load testing harness (`load_test.py`) built with `httpx` and `asyncio` to benchmark system capacity, concurrency queues, rate limiting, and response latencies across three distinct traffic profiles.

### Scenarios

1. **Steady Load (`steady`)**:
   Sends paced requests at moderate concurrency to assess nominal throughput and baseline latencies under normal conditions.
   ```bash
   python load_test.py --scenario steady --source mock
   ```

2. **Burst Load (`burst`)**:
   Dispatches a simultaneous wave of concurrent requests to test bounded queue limits, queue absorption, and immediate HTTP 503 rejection when the queue overflows.
   ```bash
   python load_test.py --scenario burst --source mock
   ```

3. **Sustained Overload (`overload`)**:
   Applies high concurrency over a continuous duration to stress sliding-window rate limiters (HTTP 429), queue timeouts, and server backpressure.
   ```bash
   python load_test.py --scenario overload --source mock --duration 5.0
   ```

4. **All Scenarios**:
   Runs all three scenarios sequentially and aggregates results into a single file.
   ```bash
   python load_test.py --scenario all --source mock --output results.csv
   ```

### Additional Configuration Options

- **Streaming / Time To First Token (TTFT)**:
  Append `--stream` to stream tokens progressively via SSE and record `first_token_ms`:
  ```bash
  python load_test.py --scenario steady --source mock --stream
  ```

- **In-Process Testing (no external server required)**:
  Append `--in-process` to run requests directly against the FastAPI ASGI app using `httpx.ASGITransport`:
  ```bash
  python load_test.py --scenario steady --in-process
  ```

- **Testing Against Real Groq LLM**:
  Set `--source groq` with modest request counts to preserve daily API quotas:
  ```bash
  python load_test.py --scenario steady --source groq --requests 3
  ```

### Output Results (`results.csv`)

Test results are saved in `results.csv` with the following 12 columns:
`request_id`, `scenario`, `source`, `start_time`, `first_token_ms`, `total_ms`, `status`, `retries`, `tokens_in`, `tokens_out`, `retrieval_setting`, `answer_grade`.

- Missing/inapplicable metrics (e.g. non-streaming TTFT, answer grade) remain empty without fabrication.
- Request statuses cleanly distinguish `success`, `rate_limited`, `queue_full`, `timeout`, and `error`.

---

## Retrieval Quality vs. Cost Experiments

The Relay includes an experimental benchmark harness (`experiment_retrieval.py`) to systematically evaluate the trade-offs between retrieval context depth (Top-K), token costs, response latency, and answer citation grounding.

### Experiment Settings

The benchmark compares three configurable retrieval settings against the exact same set of research questions:

1. **Experiment A (Top-K = 3)**:
   Minimal context depth; lowest input token cost and fastest prompt ingestion.
2. **Experiment B (Top-K = 5)**:
   Baseline default; balances passage coverage with token expenditure.
3. **Experiment C (Top-K = 8)**:
   High context coverage; captures peripheral passages but increases token cost and risks context dilution.

### How to Run Experiments

1. **Run with Mock LLM (Default, Quota Safe)**:
   Runs all 3 settings across the standard benchmark questions without consuming Groq API tokens:
   ```bash
   python experiment_retrieval.py --source mock
   ```

2. **Run with Real Groq LLM**:
   Executes the benchmark against your configured Groq model:
   ```bash
   python experiment_retrieval.py --source groq
   ```

3. **Run Specific Settings or Custom Top-K**:
   ```bash
   # Run only Experiment A and B
   python experiment_retrieval.py --settings A B --source mock

   # Custom Top-K thresholds
   python experiment_retrieval.py --top-k-a 2 --top-k-b 4 --top-k-c 6 --source mock
   ```

### Output Results (`retrieval_experiments.csv`)

Results are recorded in `retrieval_experiments.csv` capturing:
- `experiment_id`, `retrieval_setting`, `top_k`, `source`, `question`
- `retrieval_latency_ms`, `total_latency_ms`
- `tokens_in`, `tokens_out`, `total_tokens`
- `avg_similarity_score`, `min_similarity_score`, `max_similarity_score`
- `retrieved_chunk_ids`, `retrieved_papers`, `retrieved_pages`
- `is_valid`, `validation_status`, `citations_count`, `unsupported_claims_count`, `answer_preview`

At completion, the script prints an aggregated trade-off table comparing average token cost, retrieval speed, and citation validity across all three settings.

## License

This project is currently unlicensed.
