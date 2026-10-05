"""
Laya difficulty tier: classifier, resolve_tier_params, router integration,
and LogEntry plumbing. No real laya model, no checkpoint, no network.
"""
import builtins
import json
import time

import httpx
import pytest

from optmod.routing.perfrouter import laya_tier
from optmod.routing.perfrouter.laya_tier import LayaTierClassifier, TierResult, TIER_QUESTION
from optmod.routing.perfrouter.router import PerfRouterRouter, resolve_tier_params
from optmod.routing.context import RoutingContext
from optmod.schemas import ChatMessage, Features, OpenAIChatRequest, RoutingDecision

TIERS_CFG = {
    "easy":   {"delta": 0.25, "cost_cap": 2.0},
    "medium": {"delta": 0.15, "cost_cap": 2.0},
    "hard":   {"delta": 0.03, "cost_cap": 4.0},
}


def _answer(tier="hard", conf=0.8, probs=None, with_conf=True):
    probs = probs or {"easy": 0.1, "medium": 0.1, "hard": 0.8}
    ans = {"type": "choice", "choice": tier, "probabilities": probs}
    if with_conf:
        ans["answer_confidence"] = conf
    return {"answers": {"tier": ans}}


class FakeBackend:
    def __init__(self, response=None, exc=None, sleep_s=0.0):
        self.calls = []
        self.response = response if response is not None else _answer()
        self.exc = exc
        self.sleep_s = sleep_s

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.exc:
            raise self.exc
        return self.response


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _clf(backend, clock=None, **cfg):
    base = {"timeout_ms": 1000, "breaker_failures": 3, "breaker_cooldown_s": 60}
    return LayaTierClassifier({**base, **cfg}, backend=backend, clock=clock or Clock())


# ── Classifier ────────────────────────────────────────────────────────────────

def test_parses_well_formed_answer():
    be = FakeBackend(_answer("medium", 0.72, {"easy": 0.2, "medium": 0.72, "hard": 0.08}))
    res, status = _clf(be).classify("refactor this module")
    assert status == "ok"
    assert res.tier == "medium"
    assert res.confidence == pytest.approx(0.72)
    assert res.probabilities["easy"] == pytest.approx(0.2)
    assert res.latency_ms >= 0.0
    state, questions = be.calls[0]
    assert state == {"request": "refactor this module"}
    assert questions is TIER_QUESTION


def test_falls_back_to_max_probability_without_answer_confidence():
    be = FakeBackend(_answer("easy", probs={"easy": 0.66, "medium": 0.3, "hard": 0.04}, with_conf=False))
    res, status = _clf(be).classify("hi")
    assert status == "ok"
    assert res.confidence == pytest.approx(0.66)


def test_unparseable_answer_is_error():
    res, status = _clf(FakeBackend({"answers": {}})).classify("x")
    assert (res, status) == (None, "error")
    res, status = _clf(FakeBackend(_answer("trivial"))).classify("x")
    assert (res, status) == (None, "error")


def test_backend_exception_is_error_and_does_not_raise():
    res, status = _clf(FakeBackend(exc=RuntimeError("cuda oom"))).classify("x")
    assert (res, status) == (None, "error")


def test_remote_backend_is_unavailable_in_phase_1():
    clf = LayaTierClassifier({"backend": "remote", "remote": {"base_url": "http://laya.test"}})
    assert clf.classify("x") == (None, "unavailable")
    st = clf.status()
    assert st["breaker"] == "unavailable"
    assert st["target"] == "http://laya.test"
    assert "phase 2" in st["unavailable_reason"]


def test_unknown_backend_is_unavailable():
    assert LayaTierClassifier({"backend": "inprocess"}).classify("x") == (None, "unavailable")


def test_merge_config_merges_sub_blocks():
    cfg = laya_tier.merge_config({"embedded": {"device": "cuda"}, "timeout_ms": 40})
    assert cfg["embedded"] == {"checkpoint": "convaiinnovations/laya", "device": "cuda"}
    assert cfg["remote"]["base_url"] == "http://127.0.0.1:8000"
    assert cfg["timeout_ms"] == 40
    assert laya_tier.DEFAULTS["embedded"]["device"] is None   # defaults not mutated


def test_breaker_opens_and_closes_after_cooldown():
    clock = Clock()
    be = FakeBackend(exc=RuntimeError("boom"))
    clf = _clf(be, clock=clock, breaker_failures=3, breaker_cooldown_s=60)
    for i in range(3):
        assert clf.classify(f"q{i}") == (None, "error")
    assert len(be.calls) == 3

    assert clf.classify("q3") == (None, "breaker_open")
    assert len(be.calls) == 3
    assert clf.status()["breaker"] == "open"

    clock.t += 61
    be.exc = None
    res, status = clf.classify("q4")
    assert status == "ok" and res.tier == "hard"
    assert len(be.calls) == 4
    assert clf.status()["breaker"] == "closed"


def test_slow_call_returns_result_but_counts_for_breaker():
    be = FakeBackend(sleep_s=0.01)
    clf = _clf(be, timeout_ms=1, breaker_failures=2)
    res, status = clf.classify("a")
    assert status == "ok" and res is not None
    res, status = clf.classify("b")
    assert status == "ok"
    assert clf.classify("c") == (None, "breaker_open")
    assert len(be.calls) == 2


def test_memo_calls_backend_once():
    be = FakeBackend()
    clf = _clf(be)
    r1, _ = clf.classify("same text")
    r2, _ = clf.classify("same text")
    assert (r1.tier, r1.confidence) == (r2.tier, r2.confidence)
    assert r1.cached is False and r2.cached is True
    assert len(be.calls) == 1


def test_errors_are_not_memoised():
    be = FakeBackend(exc=RuntimeError("x"))
    clf = _clf(be, breaker_failures=10)
    clf.classify("t")
    clf.classify("t")
    assert len(be.calls) == 2


def test_laya_not_installed_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "laya":
            raise ImportError("No module named 'laya'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    clf = LayaTierClassifier({"backend": "embedded"})
    assert clf.classify("x") == (None, "unavailable")
    assert clf.classify("y") == (None, "unavailable")
    assert clf.status()["breaker"] == "unavailable"


def test_get_classifier_is_shared_per_config(monkeypatch):
    built = []

    class Dummy:
        def __init__(self, cfg):
            built.append(cfg)

    monkeypatch.setattr(laya_tier, "LayaTierClassifier", Dummy)
    monkeypatch.setattr(laya_tier, "_INSTANCES", {})
    a = laya_tier.get_classifier({"backend": "embedded", "mode": "shadow"})
    b = laya_tier.get_classifier({"mode": "active", "tiers": TIERS_CFG, "embedded": {}})
    c = laya_tier.get_classifier({"embedded": {"device": "cuda"}})
    assert a is b
    assert c is not a
    assert len(built) == 2


# ── resolve_tier_params ───────────────────────────────────────────────────────

LCFG = {"min_confidence": 0.5, "tiers": TIERS_CFG}


def _tr(tier="hard", conf=0.8):
    return TierResult(tier=tier, confidence=conf, probabilities={}, latency_ms=1.0)


def test_resolve_active_ok_confident_uses_tier():
    assert resolve_tier_params(_tr("hard"), "ok", "active", LCFG, 0.15, 2.0) == (0.03, 4.0, True)
    assert resolve_tier_params(_tr("easy"), "ok", "active", LCFG, 0.15, 2.0) == (0.25, 2.0, True)


@pytest.mark.parametrize("result,status,mode", [
    (_tr(conf=0.49), "ok", "active"),
    (None, "error", "active"),
    (None, "breaker_open", "active"),
    (_tr(), "ok", "shadow"),
    (_tr(), "ok", "off"),
])
def test_resolve_falls_back_to_static(result, status, mode):
    assert resolve_tier_params(result, status, mode, LCFG, 0.15, None) == (0.15, None, False)


def test_resolve_tier_null_cap_means_disabled():
    cfg = {"min_confidence": 0.5, "tiers": {"hard": {"delta": 0.02, "cost_cap": None}}}
    assert resolve_tier_params(_tr("hard"), "ok", "active", cfg, 0.15, 2.0) == (0.02, None, True)


def test_resolve_missing_tier_entry_is_static():
    cfg = {"min_confidence": 0.5, "tiers": {"easy": {"delta": 0.3}}}
    assert resolve_tier_params(_tr("hard"), "ok", "active", cfg, 0.15, 2.0) == (0.15, 2.0, False)


# ── Router integration ────────────────────────────────────────────────────────

class StubInference:
    """Stands in for PerfRouterInference: records route() kwargs."""

    def __init__(self, similarity=0.6, models=("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro")):
        self._model_ids = list(models)
        self.similarity = similarity
        self.route_calls = []
        self.classify_calls = []

    def classify_task(self, query):
        self.classify_calls.append(query)
        return [("coding.debug", self.similarity)]

    def route(self, query, **kw):
        self.route_calls.append({"query": query, **kw})
        cap = kw["cost_cap_multiplier"]
        chosen = self._model_ids[1] if cap > 2.0 else self._model_ids[0]
        return {
            "decision_model": chosen, "task_type": "coding.debug",
            "predicted_quality": 0.8, "cost_saved_pct": 0.0, "alpha": 0.3,
            "routing_mode": "normal" if self.similarity >= 0.2 else "fallback_ambiguous",
            "top_similarity": self.similarity, "pin_soft_bonus": 0.0,
        }


class _NoInference:
    def __init__(self, **kw):
        raise RuntimeError("real inference disabled in tests")


def _make_router(monkeypatch, laya_cfg=None, stub=None, backend=None):
    import sys
    import types
    import optmod.routing.perfrouter.router as rmod

    # Make both PerfRouterInference import paths fail fast — no model loads.
    fake = types.ModuleType("fake_inference")
    fake.PerfRouterInference = _NoInference
    monkeypatch.setitem(sys.modules, "perfrouter.inference.perf_router_inference", fake)
    monkeypatch.setitem(sys.modules, "optmod.routing.perfrouter.inference", fake)
    clf = _clf(backend) if backend is not None else None
    monkeypatch.setattr(rmod, "get_classifier", lambda cfg: clf)

    perf_cfg = {"degradation_threshold": 0.15, "min_similarity": 0.20, "cost_cap_multiplier": 2.0}
    if laya_cfg is not None:
        perf_cfg["laya"] = laya_cfg
    r = PerfRouterRouter({"perf_router": perf_cfg})
    r._perf_router = stub or StubInference()
    r._ready = True
    return r


def _ctx(registry, *user_msgs):
    msgs = [ChatMessage(role="user", content=m) for m in user_msgs]
    feats = Features(task_type="code", difficulty="hard", token_count=100,
                     has_tools=False, language="en", last_user_message=user_msgs[-1] if user_msgs else "")
    return RoutingContext(request=OpenAIChatRequest(messages=msgs), features=feats,
                          session_id="s", registry=registry)


ACTIVE = {"mode": "active", "min_confidence": 0.5, "tiers": TIERS_CFG}


def test_low_similarity_classifies_but_does_not_act(registry, monkeypatch):
    be = FakeBackend(_answer("hard", 0.9))
    stub = StubInference(similarity=0.05)
    r = _make_router(monkeypatch, ACTIVE, stub, be)
    d = r.route(_ctx(registry, "prove it"))
    assert len(be.calls) == 1
    assert d.meta["laya_status"] == "ok"
    assert d.meta["laya_tier"] == "hard"
    assert d.meta["pr_routing_mode"] == "fallback_ambiguous"
    assert len(stub.route_calls) == 1
    assert stub.route_calls[0]["degradation_threshold"] == 0.15
    assert stub.route_calls[0]["cost_cap_multiplier"] == 2.0
    assert d.meta["laya_applied"] is False
    assert d.meta["effective_delta"] == 0.15


def test_low_similarity_shadow_has_no_shadow_call(registry, monkeypatch):
    stub = StubInference(similarity=0.05)
    r = _make_router(monkeypatch, {**ACTIVE, "mode": "shadow"}, stub, FakeBackend(_answer("hard", 0.9)))
    d = r.route(_ctx(registry, "prove it"))
    assert len(stub.route_calls) == 1
    assert d.meta["laya_tier"] == "hard"
    assert d.meta["laya_shadow_model"] == ""


def test_empty_routing_text_skips_laya(registry, monkeypatch):
    be = FakeBackend()
    stub = StubInference()
    r = _make_router(monkeypatch, ACTIVE, stub, be)
    d = r.route(_ctx(registry, "   "))
    assert be.calls == []
    assert stub.classify_calls == []
    assert d.meta["laya_status"] == "skipped_empty"


def test_active_passes_tier_params(registry, monkeypatch):
    stub = StubInference()
    r = _make_router(monkeypatch, ACTIVE, stub, FakeBackend(_answer("hard", 0.9)))
    d = r.route(_ctx(registry, "design a distributed lock"))
    assert len(stub.route_calls) == 1
    assert stub.route_calls[0]["degradation_threshold"] == 0.03
    assert stub.route_calls[0]["cost_cap_multiplier"] == 4.0
    assert d.meta["laya_applied"] is True
    assert d.meta["laya_tier"] == "hard"
    assert d.meta["effective_delta"] == 0.03
    assert d.meta["effective_cost_cap"] == 4.0
    assert "tier=hard" in d.reason and "laya=ok" in d.reason


def test_active_low_confidence_uses_static(registry, monkeypatch):
    stub = StubInference()
    r = _make_router(monkeypatch, ACTIVE, stub, FakeBackend(_answer("hard", 0.3)))
    d = r.route(_ctx(registry, "design a distributed lock"))
    assert stub.route_calls[0]["degradation_threshold"] == 0.15
    assert d.meta["laya_status"] == "low_confidence"
    assert d.meta["laya_applied"] is False


def test_shadow_routes_static_and_records_shadow_model(registry, monkeypatch):
    stub = StubInference()
    r = _make_router(monkeypatch, {**ACTIVE, "mode": "shadow"}, stub, FakeBackend(_answer("hard", 0.9)))
    d = r.route(_ctx(registry, "design a distributed lock"))
    assert len(stub.route_calls) == 2
    assert stub.route_calls[0]["degradation_threshold"] == 0.15
    assert stub.route_calls[0]["cost_cap_multiplier"] == 2.0
    assert stub.route_calls[1]["degradation_threshold"] == 0.03
    assert stub.route_calls[1]["cost_cap_multiplier"] == 4.0
    assert d.meta["laya_shadow_model"] == "deepseek/deepseek-v4-pro"
    assert d.meta["laya_applied"] is False
    assert d.meta["effective_delta"] == 0.15


def test_shadow_skips_second_route_when_params_match(registry, monkeypatch):
    stub = StubInference()
    r = _make_router(monkeypatch, {**ACTIVE, "mode": "shadow"}, stub, FakeBackend(_answer("medium", 0.9)))
    d = r.route(_ctx(registry, "summarise this"))
    assert len(stub.route_calls) == 1
    assert d.meta["laya_shadow_model"] == ""
    assert d.meta["laya_tier"] == "medium"


def test_laya_text_has_no_repeat_and_is_left_truncated(registry, monkeypatch):
    be = FakeBackend()
    stub = StubInference()
    r = _make_router(monkeypatch, {**ACTIVE, "max_chars": 20}, stub, be)
    r.route(_ctx(registry, "first message here", "second one", "THE CURRENT TURN"))
    sent = be.calls[0][0]["request"]
    assert sent.endswith("THE CURRENT TURN")
    assert len(sent) == 20
    assert sent.count("THE CURRENT TURN") == 1
    # MiniLM text still gets the repeat
    assert stub.route_calls[0]["query"].count("THE CURRENT TURN") == 2


def test_meta_flags_cache_hit(registry, monkeypatch):
    r = _make_router(monkeypatch, ACTIVE, StubInference(), FakeBackend(_answer("medium", 0.9)))
    first = r.route(_ctx(registry, "summarise this"))
    second = r.route(_ctx(registry, "summarise this"))   # e.g. an escalation retry
    assert first.meta["laya_cached"] is False
    assert second.meta["laya_cached"] is True
    assert second.meta["laya_tier"] == "medium"


def test_meta_cached_false_when_laya_off(registry, monkeypatch):
    r = _make_router(monkeypatch, None, StubInference())
    assert r.route(_ctx(registry, "x")).meta["laya_cached"] is False


def test_laya_text_untruncated_joins_last_three(registry, monkeypatch):
    be = FakeBackend()
    r = _make_router(monkeypatch, ACTIVE, StubInference(), be)
    r.route(_ctx(registry, "a", "b", "c", "d"))
    assert be.calls[0][0]["request"] == "b\nc\nd"


def test_no_laya_block_matches_today(registry, monkeypatch):
    stub = StubInference()
    r = _make_router(monkeypatch, None, stub)
    d = r.route(_ctx(registry, "x", "y"))
    assert stub.classify_calls == []
    assert stub.route_calls == [{
        "query": "x\ny\ny", "token_count": 100, "has_images": False,
        "degradation_threshold": 0.15, "pin_info": None, "cost_cap_multiplier": 2.0,
    }]
    assert d.meta["laya_status"] == "off"
    assert r.laya_status()["mode"] == "off"


def test_null_static_cap_passes_inf(registry, monkeypatch):
    stub = StubInference()
    r = _make_router(monkeypatch, None, stub)
    r._cost_cap_multiplier = None
    d = r.route(_ctx(registry, "x"))
    assert stub.route_calls[0]["cost_cap_multiplier"] == float("inf")
    assert d.meta["effective_cost_cap"] is None


def test_set_laya_mode_and_status(registry, monkeypatch):
    r = _make_router(monkeypatch, None, StubInference(), FakeBackend())
    r.set_laya_mode("shadow")
    st = r.laya_status()
    assert st["mode"] == "shadow" and st["breaker"] == "closed"
    assert st["backend"] == "embedded" and st["target"] == "convaiinnovations/laya"
    with pytest.raises(ValueError):
        r.set_laya_mode("bogus")


def test_zero_delta_tier_warns(caplog, monkeypatch):
    with caplog.at_level("WARNING"):
        _make_router(monkeypatch, {"mode": "off", "tiers": {"hard": {"delta": 0.0}}})
    assert any("tiers.hard.delta" in r.message for r in caplog.records)


# ── LogEntry plumbing through main.chat_completions ──────────────────────────

class _MetaRouter:
    name = "stub"

    def __init__(self, registry, meta):
        self._registry, self._meta = registry, meta

    def route(self, ctx):
        return RoutingDecision(model=self._registry.primary, mutator="noop", reason="stub",
                               confidence=1.0, router_name="stub", meta=dict(self._meta))


class _OkForwarder:
    async def forward(self, model, req, msgs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, None


@pytest.fixture
def wired_main(registry, tmp_path, monkeypatch):
    from optmod import main
    from optmod.features import FeatureExtractor
    from optmod.mutators.noop import NoopMutator
    from optmod.escalation import EscalationPolicy
    from optmod.log import RoutingLog

    log_path = tmp_path / "log.jsonl"
    monkeypatch.setattr(main, "_registry", registry)
    monkeypatch.setattr(main, "_extractor", FeatureExtractor())
    monkeypatch.setattr(main, "_mutators", {"noop": NoopMutator()})
    monkeypatch.setattr(main, "_escalation", EscalationPolicy(max_escalations=0))
    monkeypatch.setattr(main, "_forwarder", _OkForwarder())
    monkeypatch.setattr(main, "_log", RoutingLog(str(log_path)))
    monkeypatch.setattr(main, "_session_pin_enabled", True)
    monkeypatch.setattr(main, "_session_pins", {})
    return main, log_path


async def _post(app, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post("/v1/chat/completions", json=body)


async def test_meta_reaches_jsonl_and_hard_pin_logs_defaults(registry, wired_main, monkeypatch):
    main, log_path = wired_main
    meta = {
        "pr_task_type": "coding.debug", "pr_routing_mode": "normal", "pr_top_similarity": 0.61,
        "laya_status": "ok", "laya_tier": "hard", "laya_confidence": 0.9, "laya_ms": 12.5,
        "laya_cached": True, "laya_applied": True, "laya_shadow_model": "", "effective_delta": 0.03,
        "effective_cost_cap": 4.0,
    }
    monkeypatch.setattr(main, "_router", _MetaRouter(registry, meta))
    body = {"messages": [{"role": "user", "content": "debug my race condition"}]}

    r1 = await _post(main.app, body)
    r2 = await _post(main.app, body)   # same session, < hard window → hard pin
    assert r1.status_code == r2.status_code == 200

    lines = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert len(lines) == 2
    first, pinned = lines
    for k, v in meta.items():
        assert first[k] == v
    assert "debug my race condition" not in log_path.read_text()

    assert pinned["pin_state"] == "hard"
    assert pinned["laya_status"] == ""
    assert pinned["laya_applied"] is False
    assert pinned["laya_cached"] is False
    assert pinned["effective_delta"] is None
    assert pinned["effective_cost_cap"] is None
