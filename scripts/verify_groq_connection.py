"""
Safe verification script for Groq API integration.
Rules:
- NEVER print, log, or expose the API key.
- Send ONLY ONE real minimal request.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 1. API key detection check
api_key = os.getenv("GROQ_API_KEY", "").strip()
has_key = bool(api_key and api_key != "your_groq_api_key_here")
print(f"API key detected: {'YES' if has_key else 'NO'}")

# 2. SDK check
sdk_installed = False
try:
    import groq
    from groq import Groq
    sdk_installed = True
    print("SDK installed: YES")
except ImportError:
    print("SDK installed: NO")
    sys.exit(1)

# 3. Client initialization check
client = None
client_initialized = False
if has_key:
    try:
        client = Groq(api_key=api_key)
        client_initialized = True
        print("Client initialized: YES")
    except Exception as e:
        print("Client initialized: NO")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        sys.exit(1)
else:
    print("Client initialized: NO")
    print("Error message: GROQ_API_KEY is not set or is placeholder")
    sys.exit(1)

# 4. Configured model
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
print(f"Model name: {model_name}")

# 5. Send ONE real minimal request
try:
    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Reply with exactly: Groq API connected successfully."}
        ],
        "max_tokens": 30,
    }
    if "gpt-oss" in model_name:
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    print("HTTP/API request result: SUCCESS")
    print(f"Response text: {content}")
except Exception as e:
    print("HTTP/API request result: FAILED")
    print(f"Error type: {type(e).__name__}")
    # Print safe error message without exposing key
    safe_err = str(e).replace(api_key, "[REDACTED]") if api_key else str(e)
    print(f"Error message: {safe_err}")
