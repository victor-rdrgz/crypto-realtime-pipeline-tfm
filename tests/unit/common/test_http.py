from datetime import UTC, datetime

import pytest
import requests

from crypto_realtime_pipeline.common.http import (
    HttpClient,
    HttpRequestError,
    RetryPolicy,
    retry_after_seconds,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text="", invalid_json=False):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client(outcomes, max_attempts=3):
    sleeps = []
    session = FakeSession(outcomes)
    http = HttpClient(
        session=session,
        retry=RetryPolicy(max_attempts=max_attempts, backoff_seconds=1.0),
        sleep=sleeps.append,
    )
    return http, session, sleeps


def test_returns_json_on_success():
    http, session, sleeps = client([FakeResponse(payload={"ok": True})])

    assert http.get_json("https://api", params={"a": 1}) == {"ok": True}
    assert session.calls == [("https://api", {"a": 1}, 30.0)]
    assert sleeps == []


def test_retries_server_errors_with_exponential_backoff():
    http, _, sleeps = client(
        [FakeResponse(503), FakeResponse(500), FakeResponse(payload=[1])], max_attempts=3
    )

    assert http.get_json("https://api") == [1]
    assert sleeps == [1.0, 2.0]


def test_retries_timeouts_and_connection_errors():
    http, _, sleeps = client(
        [requests.Timeout("slow"), requests.ConnectionError("down"), FakeResponse(payload={})]
    )

    assert http.get_json("https://api") == {}
    assert sleeps == [1.0, 2.0]


def test_respects_retry_after_on_429():
    http, _, sleeps = client(
        [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(payload={})]
    )

    http.get_json("https://api")

    assert sleeps == [7.0]


def test_fails_after_all_attempts():
    http, session, sleeps = client([FakeResponse(503)] * 3, max_attempts=3)

    with pytest.raises(HttpRequestError, match="after 3 attempts: HTTP 503"):
        http.get_json("https://api")
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_does_not_retry_client_errors():
    http, session, _ = client([FakeResponse(400, text="bad symbol")])

    with pytest.raises(HttpRequestError, match=r"HTTP 400.*bad symbol"):
        http.get_json("https://api")
    assert len(session.calls) == 1


def test_rejects_non_2xx_success_codes():
    http, session, _ = client([FakeResponse(304)])

    with pytest.raises(HttpRequestError, match="HTTP 304"):
        http.get_json("https://api")
    assert len(session.calls) == 1


def test_rejects_invalid_json():
    http, _, _ = client([FakeResponse(200, invalid_json=True)])

    with pytest.raises(HttpRequestError, match="Invalid JSON"):
        http.get_json("https://api")


def test_backoff_is_capped():
    policy = RetryPolicy(backoff_seconds=10, backoff_multiplier=10, max_backoff_seconds=60)

    assert policy.backoff(1) == 10
    assert policy.backoff(3) == 60


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("", None),
        ("12", 12.0),
        ("Wed, 21 Oct 2026 07:28:30 GMT", 30.0),
        ("Wed, 21 Oct 2026 07:27:00 GMT", 0.0),
        ("not a date", None),
    ],
)
def test_retry_after_seconds(header, expected):
    now = datetime(2026, 10, 21, 7, 28, 0, tzinfo=UTC)

    assert retry_after_seconds(header, now=now) == expected
