"""Rate limiter tracking client requests per minute using a sliding window algorithm."""

import os
import time
import asyncio
from typing import Dict, Tuple
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
DEFAULT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60.0"))


class SlidingWindowRateLimiter:
    """Tracks timestamps per client IP over a sliding window."""

    def __init__(self, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, client_id: str = "default") -> Tuple[bool, int, float]:
        """Check if client request is within quota; returns (allowed, remaining, retry_after)."""
        now = time.time()
        cutoff = now - self.window_seconds

        async with self._lock:
            timestamps = self.requests[client_id]

            # Discard timestamps outside sliding window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            current_count = len(timestamps)
            if current_count < self.max_requests:
                timestamps.append(now)
                remaining = self.max_requests - (current_count + 1)
                return True, remaining, 0.0

            oldest_ts = timestamps[0]
            retry_after = round(max(0.1, (oldest_ts + self.window_seconds) - now), 2)
            return False, 0, retry_after

    def reset(self):
        """Reset all client rate limit tracking."""
        self.requests.clear()


class ConcurrencyQueueManager:
    """Regulates maximum concurrent active requests using an asyncio Semaphore."""

    def __init__(self, max_concurrency: int = 3, queue_timeout: float = 5.0):
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.active_count = 0
        self.queued_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Wait for an execution slot up to queue_timeout seconds."""
        async with self._lock:
            self.queued_count += 1

        try:
            acquired = await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.queue_timeout,
            )
            async with self._lock:
                self.queued_count -= 1
                self.active_count += 1
            return acquired
        except asyncio.TimeoutError:
            async with self._lock:
                self.queued_count -= 1
            return False

    def release(self):
        """Release an execution slot."""
        self.semaphore.release()
        self.active_count = max(0, self.active_count - 1)

    def stats(self) -> Dict[str, int]:
        """Return active and queued count metrics."""
        return {
            "max_concurrency": self.max_concurrency,
            "active_requests": self.active_count,
            "queued_requests": self.queued_count,
        }


# Global default instances
limiter = SlidingWindowRateLimiter(max_requests=DEFAULT_MAX_REQUESTS, window_seconds=DEFAULT_WINDOW_SECONDS)
concurrency_manager = ConcurrencyQueueManager(max_concurrency=3, queue_timeout=5.0)
