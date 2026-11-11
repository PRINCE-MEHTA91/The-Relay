"""Mock LLM Service simulating real provider behaviors (latencies, errors, streaming)."""

import asyncio
import time
import uuid
import random
from typing import Dict, Any, List, Optional, AsyncGenerator
from enum import Enum


class MockMode(str, Enum):
    """Simulated LLM operational modes."""
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    FAILURE = "FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    OVERLOAD = "OVERLOAD"
    STREAMING = "STREAMING"


class MockLLMError(Exception):
    """Base exception for mock LLM errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class MockRateLimitError(MockLLMError):
    """Simulated HTTP 429 Rate Limit error."""
    def __init__(self, message: str = "Rate limit exceeded (HTTP 429). Please retry later."):
        super().__init__(message, status_code=429)


class MockServerError(MockLLMError):
    """Simulated HTTP 500 Internal Server error."""
    def __init__(self, message: str = "Internal server error (HTTP 500) from LLM provider."):
        super().__init__(message, status_code=500)


class MockTimeoutError(MockLLMError):
    """Simulated request timeout."""
    def __init__(self, message: str = "Request timed out after waiting for LLM response."):
        super().__init__(message, status_code=504)


class MockLLMService:
    """Configurable mock LLM service supporting various network and failure scenarios."""

    def __init__(
        self,
        default_mode: MockMode = MockMode.NORMAL,
        base_latency: float = 0.1,
        slow_latency: float = 1.5,
        timeout_duration: float = 2.0,
        failure_probability: float = 0.0,
        max_concurrency: int = 10,
        overload_penalty_per_request: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.default_mode = default_mode
        self.base_latency = base_latency
        self.slow_latency = slow_latency
        self.timeout_duration = timeout_duration
        self.failure_probability = failure_probability
        self.max_concurrency = max_concurrency
        self.overload_penalty_per_request = overload_penalty_per_request
        self.active_requests = 0
        self._lock = asyncio.Lock()

        if seed is not None:
            random.seed(seed)

    def _generate_mock_content(self, messages: List[Dict[str, str]]) -> str:
        """Generate a deterministic, plausible answer based on prompt keywords."""
        full_prompt = " ".join(m.get("content", "") for m in messages).lower()
        target = full_prompt.split("question:")[-1].strip() if "question:" in full_prompt else full_prompt

        # Out-of-domain unanswerable questions (negative controls)
        if any(term in target for term in ["recipe", "croissant", "quantum", "syndrome", "gpt-4o", "fault-tolerant"]):
            return "The answer cannot be found in the provided research papers."

        # Specific ground-truth benchmark questions
        if "scaled dot-product" in target or ("formula" in target and "attention" in target):
            return "Scaled dot-product attention computes outputs as Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V [Attention Is All You Need, p. 4]."
        elif "computational complexity" in target or ("per-layer" in target and "complexity" in target):
            return "Self-attention has per-layer complexity O(n^2 * d) while recurrent layers have O(n * d^2) [Attention Is All You Need, p. 6]."
        elif "multi-head attention" in target and ("subspace" in target or "combine" in target):
            return "Multi-Head Attention projects queries, keys, and values h times in parallel, concatenates and projects outputs [Attention Is All You Need, p. 4]."
        elif "transformer" in target and "architecture" in target:
            return "The Transformer uses an encoder-decoder architecture with stacked self-attention and feed-forward sublayers [Attention Is All You Need, p. 3]."
        elif "helium" in target or ("parameter count" in target and "moshi" in target) or ("base text" in target and "moshi" in target):
            return "Moshi is built on Helium, a 7B-parameter text language model built from scratch [Moshi, p. 7]."
        elif "neural audio codec" in target or "mimi" in target or "quantization" in target:
            return "Moshi uses the Mimi neural audio codec implementing split residual vector quantization (RVQ) [Moshi, p. 10]."
        elif ("user and system" in target or "audio stream" in target) and "moshi" in target:
            return "Moshi structures user and system audio as separate parallel audio codec streams [Moshi, p. 6]."
        elif "moshi" in target and ("interaction" in target or "conversational" in target or "spoken" in target):
            return "Moshi is a speech-text foundation model enabling real-time full-duplex spoken dialogue where it simultaneously listens and speaks [Moshi, p. 1]."
        elif "routing module" in target or ("boundary" in target and "module" in target):
            return "The Routing Module determines boundary positions using confidence scores in dynamic chunking [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, p. 3]."
        elif "smoothing module" in target or ("smoothing" in target and "differentiable" in target):
            return "The smoothing module smooths boundary decisions to make hierarchical chunking differentiable for end-to-end training [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, p. 3]."
        elif "width scale" in target or "monotone" in target or "d_0" in target:
            return "Model width scales monotonically across hierarchical stages D_0 <= D_1 <= ... <= D_S [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, p. 6]."
        elif "dynamic chunking" in target and "hierarchical" in target:
            return "Dynamic chunking is an adaptive segmentation method that learns variable-length chunk boundaries dynamically [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, p. 1]."
        elif "dynamic chunking" in target:
            return "Dynamic chunking segments text adaptively using boundary scoring modules rather than fixed-width splits [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, p. 7]."
        elif "moshi" in target:
            return "Moshi is a real-time speech-text foundation model enabling full-duplex spoken dialogue using separate audio codec streams [Moshi, p. 11]."
        elif "self-attention" in target:
            return "Self-attention maps query, key, and value vectors to compute attention weights via scaled dot-products [Attention Is All You Need, p. 4]."
        elif "recipe" in full_prompt or "croissant" in full_prompt or "quantum" in full_prompt:
            return "The answer cannot be found in the provided research papers."
        elif "transformer" in full_prompt and "architecture" in full_prompt:
            return "The Transformer uses an encoder-decoder architecture with stacked self-attention and feed-forward sublayers [Attention Is All You Need, p. 3]."
        elif "self-attention" in full_prompt:
            return "Self-attention maps query, key, and value vectors to compute attention weights via scaled dot-products [Attention Is All You Need, p. 4]."
        elif "moshi" in full_prompt:
            return "Moshi is a real-time speech-text foundation model enabling full-duplex spoken dialogue using separate audio codec streams [Moshi, p. 11]."
        elif "dynamic chunking" in full_prompt:
            return "Dynamic chunking segments text adaptively using boundary scoring modules rather than fixed-width splits [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, p. 7]."
        elif "recipe" in full_prompt or "croissant" in full_prompt:
            return "The answer cannot be found in the provided research papers."
        else:
            return (
                "Based on the provided research papers, the relevant concepts are discussed "
                "in the retrieved passages [Attention Is All You Need, p. 3]."
            )

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        mode: Optional[MockMode] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a mock LLM response adhering to the configured mode."""
        req_id = request_id or str(uuid.uuid4())[:8]
        active_mode = mode or self.default_mode
        start_time = time.perf_counter()

        async with self._lock:
            self.active_requests += 1
            current_active = self.active_requests

        try:
            # 1. FAILURE Mode
            if active_mode == MockMode.FAILURE:
                await asyncio.sleep(self.base_latency)
                raise MockServerError(f"Simulated server crash on request {req_id}")

            # 2. Random failure injection if configured
            if self.failure_probability > 0 and random.random() < self.failure_probability:
                await asyncio.sleep(self.base_latency)
                raise MockServerError(f"Random probabilistic failure on request {req_id}")

            # 3. RATE_LIMITED Mode
            if active_mode == MockMode.RATE_LIMITED:
                await asyncio.sleep(0.05)
                raise MockRateLimitError(f"HTTP 429: Rate limit quota exceeded on request {req_id}")

            # 4. TIMEOUT Mode
            if active_mode == MockMode.TIMEOUT:
                await asyncio.sleep(self.timeout_duration)
                raise MockTimeoutError(f"Request {req_id} timed out after {self.timeout_duration}s")

            # 5. Concurrency ceiling check
            if current_active > self.max_concurrency:
                raise MockRateLimitError(
                    f"Concurrency limit ({self.max_concurrency}) exceeded: {current_active} active"
                )

            # 6. Latency computation (NORMAL, SLOW, OVERLOAD)
            if active_mode == MockMode.SLOW:
                latency = self.slow_latency
            elif active_mode == MockMode.OVERLOAD:
                latency = self.base_latency + (current_active * self.overload_penalty_per_request)
            else:
                latency = self.base_latency

            await asyncio.sleep(latency)

            content = self._generate_mock_content(messages)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            in_tokens = sum(len(m.get("content", "").split()) for m in messages)
            out_tokens = len(content.split())

            return {
                "content": content,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "latency_ms": elapsed_ms,
                "model": f"mock-{active_mode.value.lower()}",
                "status": "success",
                "request_id": req_id,
                "error": None,
            }

        except MockLLMError as err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": elapsed_ms,
                "model": f"mock-{active_mode.value.lower()}",
                "status": "error",
                "request_id": req_id,
                "error": str(err),
                "status_code": err.status_code,
            }
        finally:
            async with self._lock:
                self.active_requests -= 1

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        chunk_delay: float = 0.03,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Simulate progressive token streaming through an async generator."""
        req_id = request_id or str(uuid.uuid4())[:8]
        content = self._generate_mock_content(messages)
        words = content.split(" ")

        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(chunk_delay)
            yield token


# Global singleton instance
mock_service = MockLLMService()

