"""Task 8: Groq API client with retry handling, exponential backoff, and metrics."""

import os
import time
import random
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import groq
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Config defaults
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))


def get_groq_client() -> Groq:
    """Initialize and return the Groq client using API key from .env."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or is still the placeholder. "
            "Please add your valid Groq API key to .env file."
        )
    return Groq(api_key=api_key)


def get_available_models(client: Optional[Groq] = None) -> List[str]:
    """Fetch the list of currently active model IDs from Groq."""
    if client is None:
        try:
            client = get_groq_client()
        except ValueError:
            return []
    model_list = client.models.list()
    return [m.id for m in model_list.data if getattr(m, "active", True)]


def send_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    client: Optional[Groq] = None,
) -> Dict[str, Any]:
    """
    Send messages to Groq with retries, exponential backoff, and latency tracking.
    Handles rate limits (429), timeouts, connection errors, and server errors.
    """
    if client is None:
        try:
            client = get_groq_client()
        except ValueError as ve:
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0.0,
                "model": model or DEFAULT_MODEL,
                "status": "error",
                "error": str(ve),
            }

    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    temp = DEFAULT_TEMPERATURE if temperature is None else temperature
    tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    start_time = time.perf_counter()
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

            return {
                "content": response.choices[0].message.content or "",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": elapsed_ms,
                "model": selected_model,
                "status": "success",
                "error": None,
            }

        except groq.AuthenticationError as e:
            # Fatal: invalid API key, retrying will not help
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": elapsed_ms,
                "model": selected_model,
                "status": "error",
                "error": f"Authentication failed: Check your GROQ_API_KEY in .env. ({e})",
            }

        except (groq.NotFoundError, groq.BadRequestError) as e:
            # Fatal: invalid model name or prompt structure
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": elapsed_ms,
                "model": selected_model,
                "status": "error",
                "error": f"Request rejected ({type(e).__name__}): {e}",
            }

        except (groq.RateLimitError, groq.InternalServerError, groq.APITimeoutError, groq.APIConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff with small random jitter
                backoff_delay = (initial_delay * (2 ** attempt)) + random.uniform(0.1, 0.5)
                error_type = "Rate limit hit (429)" if isinstance(e, groq.RateLimitError) else type(e).__name__
                print(f"  ⚠️  {error_type}: Retry {attempt + 1}/{max_retries} in {backoff_delay:.2f}s...")
                time.sleep(backoff_delay)
            else:
                break
        except Exception as e:
            last_error = e
            break

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "content": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": elapsed_ms,
        "model": selected_model,
        "status": "error",
        "error": f"Request failed after {max_retries} retries ({type(last_error).__name__}): {last_error}",
    }


def main():
    """Verify Groq API connectivity and list available models."""
    print("=" * 62)
    print("  THE RELAY — Task 8: Groq API Integration Test")
    print("=" * 62)

    try:
        client = get_groq_client()
        print("🔍 Connecting to Groq and fetching available models...")
        models = get_available_models(client)
        print(f"✅ Connection successful! Found {len(models)} models.")
        print(f"Top models: {', '.join(models[:4])}")

        test_model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        print(f"\n🧪 Sending test ping using model: {test_model}")
        res = send_chat_completion(
            messages=[{"role": "user", "content": "Respond with 'The Relay is online' in 5 words or fewer."}],
            model=test_model,
            max_tokens=30,
            client=client,
        )

        if res["status"] == "success":
            print(f"✅ Response received in {res['latency_ms']} ms")
            print(f"  Content: {res['content'].strip()}")
            print(f"  Tokens : {res['input_tokens']} in / {res['output_tokens']} out")
        else:
            print(f"❌ Groq ping error: {res['error']}")

    except ValueError as ve:
        print(f"⚠️  Configuration warning: {ve}")
    except Exception as e:
        print(f"❌ Error testing Groq service: {e}")


if __name__ == "__main__":
    main()
