"""LLM client wrapper.

The only file that knows which model vendor we use. Nodes depend on the
LLMClient protocol below, not on any SDK -- swapping providers means adding
a class here and changing one line in cli.py.
"""

import re
import threading
import time
from typing import Protocol, TypeVar

from google import genai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.6-flash"

# Gemini's free tier allows 5 requests/minute per model. Paid tiers are far
# higher -- raise this if you have quota.
DEFAULT_REQUESTS_PER_MINUTE = 5

_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


class LLMError(Exception):
    """Raised when the model call fails or returns unusable output.

    `retry_after` carries the server's requested wait in seconds when it sent
    one, so the caller can back off for exactly as long as it was told to
    rather than guessing.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


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
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            time.sleep(wait)


class LLMClient(Protocol):
    """What the nodes need from a language model. Nothing else."""

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Return an instance of `schema` produced from `prompt`."""
        ...


class GeminiClient:
    """LLMClient backed by the Google Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._limiter = RateLimiter(requests_per_minute)

    def generate(self, prompt: str, schema: type[T]) -> T:
        """Call Gemini in JSON mode with `schema` as the response schema.

        Raises LLMError if the call fails or the response cannot be parsed
        into `schema`.
        """
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
            raise LLMError(
                f"{self._model} call failed: {_summarize_error(exc)}",
                retry_after=_retry_after(exc),
            ) from exc

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
