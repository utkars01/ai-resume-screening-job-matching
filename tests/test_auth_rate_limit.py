from collections import defaultdict, deque
from threading import Lock
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.app.api import auth as auth_api


def test_login_rate_limit_returns_retry_after_and_resets_after_window(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(auth_api, "_attempts", defaultdict(deque))
    monkeypatch.setattr(auth_api, "_attempts_lock", Lock())
    monkeypatch.setattr(auth_api.time, "monotonic", lambda: now[0])
    request = SimpleNamespace(client=SimpleNamespace(host="test-client"))

    for _ in range(auth_api._LOGIN_LIMIT):
        auth_api._rate_limit(request, registering=False)
    with pytest.raises(HTTPException) as error:
        auth_api._rate_limit(request, registering=False)

    assert error.value.status_code == 429
    assert int(error.value.headers["Retry-After"]) > 0
    assert error.value.detail == "Too many account requests. Try again later."

    now[0] += auth_api._LOGIN_WINDOW_SECONDS
    auth_api._rate_limit(request, registering=False)


def test_registration_and_login_have_separate_rate_limit_buckets(monkeypatch):
    monkeypatch.setattr(auth_api, "_attempts", defaultdict(deque))
    monkeypatch.setattr(auth_api, "_attempts_lock", Lock())
    monkeypatch.setattr(auth_api.time, "monotonic", lambda: 200.0)
    request = SimpleNamespace(client=SimpleNamespace(host="test-client"))

    for _ in range(auth_api._REGISTER_LIMIT):
        auth_api._rate_limit(request, registering=True)
    with pytest.raises(HTTPException) as error:
        auth_api._rate_limit(request, registering=True)

    assert error.value.status_code == 429
    # Exhausting the registration bucket does not consume the login bucket.
    auth_api._rate_limit(request, registering=False)
