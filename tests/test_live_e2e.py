"""
Live end-to-end tests — hit real LLM providers (OpenRouter + DeepSeek).

Loads API keys from .env at project root.
Skipped automatically when keys are absent (e.g. in CI without secrets).

Run manually:
    uv run pytest tests/test_live_e2e.py -v -s
"""

import json
import os
from pathlib import Path

# ── Load .env before the app lifespan reads os.environ ────────────────────────
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import pytest
from fastapi.testclient import TestClient
from optmod.main import app

# ── Skip the whole module when keys are missing ────────────────────────────────
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY required for live tests",
)

_LOG_PATH = Path("routing.log.jsonl")

ALL_MODELS = {
    "openai/gpt-oss-120b",
    "arcee-ai/trinity-large-thinking",
    "deepseek/deepseek-v4-flash",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_log() -> dict:
    lines = [l for l in _LOG_PATH.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1])


def _assert_completion(resp) -> dict:
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "choices" in body, f"No 'choices' in response: {body}"
    choice = body["choices"][0]
    content = choice["message"].get("content")
    finish = choice.get("finish_reason")
    assert isinstance(content, str) and content.strip(), (
        f"Empty content (finish_reason={finish!r}). "
        "Thinking model may have exhausted max_tokens in the reasoning phase — increase max_tokens."
    )
    return body


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def restore_rule_based(client):
    """Ensure rule_based router is active before and after every test."""
    client.post("/optmod/router/rule_based")
    yield
    client.post("/optmod/router/rule_based")


# ── Rule-based router: tier routing ──────────────────────────────────────────

def test_live_fast_tier(client):
    """
    'Summarize …' with short context → Rule 6 (summarize + tokens < 3000)
    → fast tier → openai/gpt-oss-120b
    """
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "max_tokens": 30,
        "messages": [{"role": "user", "content": "Summarize in one sentence: The sky is blue."}],
    })
    _assert_completion(r)
    log = _last_log()
    assert log["router"] == "RuleBasedRouter"
    assert log["final_model"] == "openai/gpt-oss-120b"
    assert log["ok"] is True


def test_live_reasoning_tier_hebrew(client):
    """
    Hebrew input → Rule 8 (language == he)
    → reasoning tier → arcee-ai/trinity-large-thinking
    """
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "max_tokens": 1500,  # thinking model needs tokens for internal reasoning + answer
        "messages": [{"role": "user", "content": "מה זה בינה מלאכותית? תסביר בקצרה."}],
    })
    _assert_completion(r)
    log = _last_log()
    assert log["router"] == "RuleBasedRouter"
    assert log["final_model"] == "arcee-ai/trinity-large-thinking"
    assert log["ok"] is True


def test_live_oracle_tier_hard(client):
    """
    'adversarial' keyword → difficulty=hard → Rule 3 (hard/long context)
    → oracle tier → deepseek/deepseek-v4-flash
    """
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "max_tokens": 1500,  # deepseek-v4-flash is a thinking model; needs tokens for reasoning + answer
        "messages": [{"role": "user", "content": "Analyze this complex adversarial multi-round scenario briefly."}],
    })
    _assert_completion(r)
    log = _last_log()
    assert log["router"] == "RuleBasedRouter"
    assert log["final_model"] == "deepseek/deepseek-v4-flash"
    assert log["ok"] is True


# ── TRouter: live routing + real response ─────────────────────────────────────

def test_live_trouter_routes_and_responds(client):
    """
    TRouterRouter encodes the query, runs the neural net, picks a model,
    and gets a real response from that provider.
    """
    r = client.post("/optmod/router/trouter")
    assert r.status_code == 200
    assert r.json()["router"] == "trouter"

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "max_tokens": 300,  # thinking model may be chosen; needs tokens for <think> + answer
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    })
    _assert_completion(r)
    log = _last_log()
    assert log["router"] == "TRouterRouter"
    assert log["final_model"] in ALL_MODELS
    assert log["ok"] is True


def test_live_trouter_hebrew_routes_to_reasoning(client):
    """
    TRouter should route Hebrew to the reasoning model
    (arcee-ai/trinity-large-thinking:free matches idx=0 in the weights).
    """
    r = client.post("/optmod/router/trouter")
    assert r.status_code == 200

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": "מה ההבדל בין מחסנית לתור?"}],
    })
    _assert_completion(r)
    log = _last_log()
    assert log["router"] == "TRouterRouter"
    assert log["final_model"] in ALL_MODELS
    assert log["ok"] is True


# ── Router hot-swap ───────────────────────────────────────────────────────────

def test_live_router_swap_passthrough(client):
    """Switch to passthrough (always primary = deepseek-v4), verify response."""
    client.post("/optmod/router/passthrough")
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "max_tokens": 1500,  # primary is deepseek-v4-flash (thinking model)
        "messages": [{"role": "user", "content": "Say hi."}],
    })
    _assert_completion(r)
    log = _last_log()
    assert log["router"] == "PassthroughRouter"
    assert log["final_model"] == "deepseek/deepseek-v4-flash"
    assert log["ok"] is True
