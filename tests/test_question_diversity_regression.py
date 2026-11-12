"""Regression test suite ensuring question-specific answers, distinct mock/Groq outputs, and refusal handling."""

import os
import sys
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from core.mock_llm import MockMode, mock_service
from core.groq_service import get_groq_client

client = TestClient(app)


def test_mock_mode_question_diversity_and_refusals():
    """Verify that different questions do NOT return the same hardcoded answer in Mock mode."""
    q_moshi = "What is Moshi and how does it support real-time dialogue?"
    q_transformer = "What is the Transformer architecture?"
    q_attention = "Explain multi-head attention."
    q_rag = "What is RAG?"
    q_bitcoin = "What is the current price of Bitcoin?"

    res_moshi = client.post("/ask", json={"question": q_moshi, "use_mock": True}).json()
    res_transformer = client.post("/ask", json={"question": q_transformer, "use_mock": True}).json()
    res_attention = client.post("/ask", json={"question": q_attention, "use_mock": True}).json()
    res_rag = client.post("/ask", json={"question": q_rag, "use_mock": True}).json()
    res_bitcoin = client.post("/ask", json={"question": q_bitcoin, "use_mock": True}).json()

    # 1. Assert each answer is distinct and relevant
    assert res_moshi["answer"] != res_transformer["answer"], "Moshi and Transformer returned the same answer!"
    assert res_moshi["answer"] != res_attention["answer"], "Moshi and Attention returned the same answer!"
    assert res_moshi["answer"] != res_rag["answer"], "Moshi and RAG returned the same answer!"
    assert res_transformer["answer"] != res_attention["answer"], "Transformer and Attention returned the same answer!"

    # 2. Assert question-specific content
    assert "Moshi" in res_moshi["answer"]
    assert "Transformer" in res_transformer["answer"]
    assert "Multi-Head Attention" in res_attention["answer"]

    # 3. Assert out-of-corpus questions are refused
    assert "cannot be found in the provided research papers" in res_rag["answer"]
    assert res_rag["status"] == "REFUSAL_VERIFIED"
    assert res_rag["is_valid"] is True

    assert "cannot be found in the provided research papers" in res_bitcoin["answer"]
    assert res_bitcoin["status"] == "REFUSAL_VERIFIED"
    assert res_bitcoin["is_valid"] is True

    # 4. Assert in-corpus questions are verified
    assert res_moshi["is_valid"] is True
    assert res_moshi["status"] == "verified"
    assert res_transformer["is_valid"] is True
    assert res_transformer["status"] == "verified"


def test_mock_streaming_question_diversity():
    """Verify that /ask/stream emits distinct progressive tokens for different questions."""
    with client.stream("POST", "/ask/stream", json={"question": "What is Moshi?", "use_mock": True}) as r1:
        tokens_moshi = []
        for line in r1.iter_lines():
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                payload = json.loads(line_str[6:])
                if payload.get("token"):
                    tokens_moshi.append(payload["token"])

    with client.stream("POST", "/ask/stream", json={"question": "What is RAG?", "use_mock": True}) as r2:
        tokens_rag = []
        for line in r2.iter_lines():
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                payload = json.loads(line_str[6:])
                if payload.get("token"):
                    tokens_rag.append(payload["token"])

    moshi_stream = "".join(tokens_moshi)
    rag_stream = "".join(tokens_rag)

    assert moshi_stream != rag_stream, "Streaming returned identical content for Moshi and RAG!"
    assert "Moshi" in moshi_stream
    assert "cannot be found in the provided research papers" in rag_stream


def test_groq_mode_configuration_and_diversity():
    """Verify live Groq API integration and distinct live responses when GROQ_API_KEY is configured."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key or groq_key == "your_groq_api_key_here":
        pytest.skip("GROQ_API_KEY is not configured; skipping live Groq API regression test.")

    # Single-test live verification for question diversity
    res_groq_a = client.post(
        "/ask",
        json={"question": "What is the Transformer architecture?", "use_mock": False, "top_k": 3},
    ).json()

    res_groq_d = client.post(
        "/ask",
        json={"question": "What is RAG?", "use_mock": False, "top_k": 3},
    ).json()

    assert res_groq_a["provider"] == "groq"
    assert res_groq_d["provider"] == "groq"
    assert res_groq_a["answer"] != res_groq_d["answer"]
    assert "Transformer" in res_groq_a["answer"] or "encoder" in res_groq_a["answer"].lower()
    assert "cannot be found in the provided research papers" in res_groq_d["answer"]
    assert res_groq_a["is_valid"] is True
    assert res_groq_d["is_valid"] is True
