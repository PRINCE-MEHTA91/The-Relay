"""
Test /ask endpoint with live Groq service (use_mock=False).
Send ONLY ONE request.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

if __name__ == "__main__":
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "What is the Transformer architecture?",
            "use_mock": False,
            "top_k": 3,
        },
    )

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Status: {data.get('status')}")
    print(f"Answer: {data.get('answer')[:180]}...")
    print(f"Input tokens: {data.get('input_tokens')}")
    print(f"Output tokens: {data.get('output_tokens')}")
    print(f"Citations count: {len(data.get('citations', []))}")
    print(f"Latency ms: {data.get('latency_ms')}")

