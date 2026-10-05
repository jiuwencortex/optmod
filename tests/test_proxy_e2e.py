import os

import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from optmod.main import app

# All provider base URLs that optmod might route to
_PROVIDERS = [
    "https://openrouter.ai/api/v1",
    "https://api.deepseek.com",
    "https://generativelanguage.googleapis.com/v1beta/openai",
    "https://api.anthropic.com/v1",
    "https://api.openai.com/v1",
    "https://api.moonshot.cn/v1",
]

OK_RESPONSE = {
    "id": "test", "object": "chat.completion", "created": 1,
    "model": "test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


_KEY_ENVS = ["OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY",
             "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY"]


@pytest.fixture(scope="module")
def client():
    # Models whose native provider is not in config.yaml fall back to OpenRouter,
    # which needs a key at startup. Dummy keys keep the suite independent of .env;
    # real keys, if present, are left alone. Nothing here touches the network.
    mp = pytest.MonkeyPatch()
    for env in _KEY_ENVS:
        if not os.environ.get(env):
            mp.setenv(env, "test-key")
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mp.undo()


@respx.mock
def test_simple_route_succeeds(client):
    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(
            return_value=httpx.Response(200, json=OK_RESPONSE)
        )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "summarize this briefly"}],
    })
    assert r.status_code == 200


@respx.mock
def test_escalation_on_429(client):
    """First chosen model returns 429; proxy escalates and the next attempt succeeds."""
    counter = {"n": 0}

    def respond(request):
        counter["n"] += 1
        if counter["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=OK_RESPONSE)

    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(side_effect=respond)

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 200


@respx.mock
def test_all_models_fail_returns_502(client):
    """All models fail → 502."""
    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(
            return_value=httpx.Response(500)
        )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 502


@respx.mock
def test_auth_error_no_escalation(client):
    """401 is non-retryable → 502 immediately, no escalation."""
    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(
            return_value=httpx.Response(401)
        )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 502
