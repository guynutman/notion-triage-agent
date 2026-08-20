"""Rate limiter and error-summary tests.

GeminiClient itself needs the network, but the pieces that decide *when* to
call and *what to report* are pure and testable.
"""

import threading
import time

from notion_triage_agent.llm import LLMError, RateLimiter, _retry_after, _summarize_error


def test_rate_limiter_spaces_calls_out():
    limiter = RateLimiter(per_minute=600)  # 100ms apart
    started = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - started >= 0.2


def test_rate_limiter_is_shared_across_threads():
    """The thread pool must not each get their own quota."""
    limiter = RateLimiter(per_minute=600)
    started = time.monotonic()
    threads = [threading.Thread(target=limiter.acquire) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert time.monotonic() - started >= 0.3


def test_rate_limiter_disabled_by_zero():
    limiter = RateLimiter(per_minute=0)
    started = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - started < 0.01


def test_error_summary_extracts_the_message_from_a_json_blob():
    raw = Exception(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
        "your current quota', 'status': 'RESOURCE_EXHAUSTED'}}"
    )
    assert _summarize_error(raw) == "You exceeded your current quota"


def test_error_summary_truncates_long_text():
    assert len(_summarize_error(Exception("x" * 500))) <= 163


def test_retry_delay_is_read_from_the_error():
    assert _retry_after(Exception("... 'retryDelay': '41.9s' ...")) == 41.9
    assert _retry_after(Exception("no hint here")) is None


def test_llm_error_carries_no_delay_by_default():
    assert LLMError("boom").retry_after is None
