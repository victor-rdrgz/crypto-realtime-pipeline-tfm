import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpRequestError(RuntimeError):
    """A request that failed permanently or after all retries."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0

    def backoff(self, attempt: int) -> float:
        delay = self.backoff_seconds * self.backoff_multiplier ** (attempt - 1)
        return min(delay, self.max_backoff_seconds)


def retry_after_seconds(value: str | None, now: datetime | None = None) -> float | None:
    """Parses a Retry-After header given in seconds or as an HTTP date."""
    if not value:
        return None
    if value.strip().isdigit():
        return float(value)
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    reference = now or datetime.now(UTC)
    return max(0.0, (moment - reference).total_seconds())


class HttpClient:
    """JSON-over-HTTP client with retries on timeouts, connection errors, 429 and 5xx."""

    def __init__(
        self,
        session: requests.Session | None = None,
        retry: RetryPolicy | None = None,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session = session or requests.Session()
        self._retry = retry or RetryPolicy()
        self._timeout = timeout_seconds
        self._sleep = sleep

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        last_error: str = ""
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                wait = self._retry.backoff(attempt)
            else:
                if 200 <= response.status_code < 300:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise HttpRequestError(f"Invalid JSON from {url}") from exc
                if response.status_code not in RETRYABLE_STATUS:
                    raise HttpRequestError(
                        f"HTTP {response.status_code} from {url}: {response.text[:500]}"
                    )
                last_error = f"HTTP {response.status_code}"
                wait = retry_after_seconds(response.headers.get("Retry-After"))
                if wait is None:
                    wait = self._retry.backoff(attempt)
            if attempt < self._retry.max_attempts:
                self._sleep(wait)
        raise HttpRequestError(
            f"{url} failed after {self._retry.max_attempts} attempts: {last_error}"
        )
