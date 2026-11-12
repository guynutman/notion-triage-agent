"""LLM client wrapper.

The only file that knows which model vendor we use. Nodes depend on the
LLMClient protocol in models.py, not on this file or any SDK -- swapping
providers means adding a class here and changing one line in cli.py.
"""

import re
import threading
import time
from typing import TypeVar

from google import genai
from pydantic import BaseModel

from notion_triage_agent.models import LLMClient, LLMError

__all__ = ["DEFAULT_MODEL", "GeminiClient", "LLMClient", "LLMError", "RateLimiter"]

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.6-flash"

# Gemini's free tier allows 5 requests/minute per model. Paid tiers are far
# higher -- raise this if you have quota.
DEFAULT_REQUESTS_PER_MINUTE = 5

# A requested retry delay longer than this means the daily quota is spent.
QUOTA_EXHAUSTED_SECONDS = 300.0

_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


class RateLimiter:
    """Spaces calls out to at most `per_minute` of them, across all threads.

    The analysis nodes fan out over a thread pool, so without this the pool
    would burn a whole minute's quota in its first second and every worker
    would come back 429.
    """

    def __init__(self, per_minute: int):
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        """Block until this thread is allowed to make its call."""
        if not self._interval:
            return
        with self._lock:
            slot = max(time.monotonic(), self._next_slot)
            self._next_slot = slot + self._interval
        # Loop rather than sleep once: on Windows time.sleep can return a few
        # milliseconds early, which would let calls bunch up past the limit.
        while (remaining := slot - time.monotonic()) > 0:
            time.sleep(remaining)


class GeminiClient:
    """LLMClient (see models.py) backed by the Google Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._limiter = RateLimiter(requests_per_minute)
        # Set once the server says the quota is exhausted for longer than a
        # retry is worth. Every later call then fails instantly instead of
        # spending a rate-limit slot just to be told the same thing.
        self._exhausted: LLMError | None = None

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Call Gemini in JSON mode with `schema` as the response schema.

        Raises LLMError if the call fails or the response cannot be parsed
        into `schema`.
        """
        if self._exhausted is not None:
            raise self._exhausted
        self._limiter.acquire()
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
        except Exception as exc:  # network, auth, quota, safety blocks
            error = LLMError(
                f"{self._model} call failed: {_summarize_error(exc)}",
                retry_after=_retry_after(exc),
            )
            if error.retry_after is not None and error.retry_after > QUOTA_EXHAUSTED_SECONDS:
                hours = error.retry_after / 3600
                self._exhausted = LLMError(
                    f"{self._model} quota exhausted (resets in ~{hours:.1f} h); "
                    f"try another model with --model"
                )
                raise self._exhausted from exc
            raise error from exc

        parsed = response.parsed
        if not isinstance(parsed, schema):
            raise LLMError(
                f"{self._model} returned output that did not match "
                f"{schema.__name__}: {response.text!r}"
            )
        return parsed


def _summarize_error(exc: Exception) -> str:
    """Condense an SDK exception into one readable line.

    Google's errors arrive as a full JSON quota report; printing all of it
    buries the ten useful characters under four hundred noisy ones.
    """
    text = str(exc)
    match = re.search(r"'message':\s*'([^']+)'", text)
    if match:
        text = match.group(1)
    text = " ".join(text.split())
    return text[:160] + ("..." if len(text) > 160 else "")


def _retry_after(exc: Exception) -> float | None:
    """Pull the server's requested retry delay out of an error, if present."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None
