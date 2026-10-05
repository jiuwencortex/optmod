import hashlib
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from optmod.config import load_config, Config
from optmod.registry import ModelRegistry, ModelConfig
from optmod.features import FeatureExtractor
from optmod.routing import BaseRouter, build_router
from optmod.mutators import BaseContextMutator
from optmod.mutators.noop import NoopMutator
from optmod.mutators.thinking_mode import ThinkingModeMutator
from optmod.mutators.tool_result_compressor import ToolResultCompressorMutator
from optmod.escalation import EscalationPolicy
from optmod.forwarder import ModelForwarder
from optmod.log import RoutingLog
from optmod.schemas import OpenAIChatRequest, RoutingContext, RoutingDecision, LogEntry, SessionPin

# RoutingDecision.meta keys copied verbatim into LogEntry
_META_LOG_FIELDS = (
    "pr_task_type", "pr_routing_mode", "pr_top_similarity",
    "laya_status", "laya_tier", "laya_confidence", "laya_ms", "laya_cached",
    "laya_applied", "laya_shadow_model", "effective_delta", "effective_cost_cap",
)
from optmod.stats import stats_router

_registry:             ModelRegistry       | None = None
_extractor:            FeatureExtractor    | None = None
_router:               BaseRouter          | None = None
_mutators:             dict[str, BaseContextMutator] = {}
_escalation:           EscalationPolicy    | None = None
_forwarder:            ModelForwarder      | None = None
_log:                  RoutingLog          | None = None
_config:               Config              | None = None
_tool_compressor_on:   bool               = False
_session_pin_enabled:  bool               = True
_session_pins:         dict[str, SessionPin] = {}
_hard_window_s:        float              = 300.0
_soft_window_s:        float              = 1800.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _extractor, _router, _mutators
    global _escalation, _forwarder, _log, _config, _tool_compressor_on
    global _hard_window_s, _soft_window_s, _session_pin_enabled

    _config     = load_config("config.yaml")
    _tool_compressor_on = bool(_config.dict().get("tool_result_compressor", False))
    pin_cfg = _config._raw.get("session_pin", {})
    _hard_window_s = float(
        pin_cfg.get("hard_window_s") or _config._raw.get("session_pin_hard_window_s", 300.0)
    )
    _soft_window_s = float(
        pin_cfg.get("soft_window_s") or _config._raw.get("session_pin_soft_window_s", 1800.0)
    )
    _session_pin_enabled = bool(_config.dict().get("session_pin_enabled", True))
    _session_pins.clear()
    _registry   = ModelRegistry(_config.models, _config.primary_model)
    _extractor  = FeatureExtractor()
    _router     = build_router(_config.router, _config.dict())
    _mutators   = {
        "noop":                   NoopMutator(),
        "thinking_mode":          ThinkingModeMutator(),
        "tool_result_compressor": ToolResultCompressorMutator(),
    }
    _escalation = EscalationPolicy(max_escalations=_config.escalation.max_escalations)
    _forwarder  = ModelForwarder()
    _log        = RoutingLog(_config.log_path)

    yield

    await _forwarder.close()
    _log.flush()


app = FastAPI(title="optmod", lifespan=lifespan)
app.include_router(stats_router)
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


def _derive_session_id(req: OpenAIChatRequest) -> str:
    first = next(
        (m.content for m in req.messages if m.role == "user" and isinstance(m.content, str)),
        str(id(req)),
    )
    return hashlib.sha1(str(first).encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _completion_to_sse(resp: dict) -> Generator[str, None, None]:
    """Wrap a non-streaming chat.completion response in SSE so streaming clients work."""
    msg_id  = resp.get("id", "")
    created = resp.get("created", 0)
    model   = resp.get("model", "")
    usage   = resp.get("usage", {})

    choices = resp.get("choices", [])
    if choices:
        choice       = choices[0]
        message      = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls   = message.get("tool_calls")

        delta: dict = {"role": message.get("role", "assistant")}
        if tool_calls:
            delta["content"]    = None
            delta["tool_calls"] = [
                {"index": i, **{k: v for k, v in tc.items() if k != "index"}}
                for i, tc in enumerate(tool_calls)
            ]
        else:
            delta["content"] = message.get("content") or ""

        # Propagate non-standard fields (reasoning, etc.) so clients see them
        for key in ("reasoning", "reasoning_details"):
            if key in message:
                delta[key] = message[key]

        chunk = {"id": msg_id, "object": "chat.completion.chunk",
                 "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"

        final = {"id": msg_id, "object": "chat.completion.chunk",
                 "created": created, "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                 "usage": usage}
        yield f"data: {json.dumps(final)}\n\n"

    yield "data: [DONE]\n\n"


def _request_has_images(req: OpenAIChatRequest) -> bool:
    for m in req.messages:
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


def _get_active_pin(session_id: str, now: float) -> SessionPin | None:
    pin = _session_pins.get(session_id)
    if pin is None:
        return None
    if now - pin.last_turn_at > _soft_window_s:
        _session_pins.pop(session_id, None)
        return None
    return pin


def _update_pin(session_id: str, model_name: str, cached_tokens: int, prompt_tokens: int, now: float) -> None:
    existing   = _session_pins.get(session_id)
    turn_count = (existing.turn_count + 1) if existing else 1
    cache_rate = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0
    _session_pins[session_id] = SessionPin(
        model_name=         model_name,
        last_turn_at=       now,
        last_cache_rate=    cache_rate,
        last_prompt_tokens= prompt_tokens,
        turn_count=         turn_count,
    )


def _rehome_pin(pin_model: ModelConfig, ctx: RoutingContext) -> ModelConfig | None:
    """Pick a larger-context replacement closest to pin_model in tier and cost."""
    needed = ctx.features.token_count
    candidates = [
        m for m in ctx.registry.all()
        if m.name != pin_model.name
        and m.context_window >= needed
        and (m.supports_tools or not pin_model.supports_tools)
        and (m.supports_vision or not pin_model.supports_vision)
    ]
    if not candidates:
        return None
    same_tier = [m for m in candidates if m.tier == pin_model.tier]
    pool      = same_tier or candidates
    return min(pool, key=lambda m: abs(m.cost_per_1k - pin_model.cost_per_1k))


def _try_hard_pin(
    pin:  SessionPin,
    ctx:  RoutingContext,
    now:  float,
) -> tuple[RoutingDecision | None, str]:
    """Return (decision, pin_state). decision=None means evict and fall through to router."""
    if pin.model_name not in {m.name for m in ctx.registry.all()}:
        return None, "evicted_missing_model"
    pin_model = ctx.registry.get(pin.model_name)
    if _request_has_images(ctx.request) and not pin_model.supports_vision:
        return None, "evicted_vision"
    if ctx.features.token_count > pin_model.context_window:
        replacement = _rehome_pin(pin_model, ctx)
        if replacement is None:
            return None, "evicted_context"
        pin.model_name = replacement.name
        return RoutingDecision(
            model=       replacement,
            mutator=     "noop",
            reason=      f"session_pin_rehomed from={pin_model.name} to={replacement.name}",
            confidence=  1.0,
            router_name= "session_pin",
        ), "rehomed_context"
    return RoutingDecision(
        model=       pin_model,
        mutator=     "noop",
        reason=      f"session_pin_hard age_s={int(now - pin.last_turn_at)} turn={pin.turn_count}",
        confidence=  1.0,
        router_name= "session_pin",
    ), "hard"


@app.post("/v1/chat/completions")
async def chat_completions(raw: Request) -> JSONResponse:
    body     = await raw.json()
    req      = OpenAIChatRequest(**body)
    features = _extractor.extract(req)
    session  = _derive_session_id(req)
    t0       = time.perf_counter()
    now      = time.time()

    ctx = RoutingContext(
        request=req,
        features=features,
        session_id=session,
        registry=_registry,
    )

    # ── Session-pin gate ────────────────────────────────────────────────
    pin = _get_active_pin(session, now) if _session_pin_enabled else None
    pinned_decision: RoutingDecision | None = None
    pin_state = "fresh"

    if pin is not None:
        age = now - pin.last_turn_at
        if age < _hard_window_s:
            pinned_decision, pin_state = _try_hard_pin(pin, ctx, now)
            if pinned_decision is None:
                _session_pins.pop(session, None)
                pin = None
        else:
            pin_state = "soft"
    ctx.session_pin = pin

    response:   dict             = {}
    error_type: str | None       = None
    decision:   RoutingDecision | None = None

    for attempt in range(_escalation.max_escalations + 1):
        ctx.attempt_number = attempt

        if attempt == 0 and pinned_decision is not None:
            decision = pinned_decision
        else:
            decision = _router.route(ctx)
        mutator  = _mutators.get(decision.mutator, _mutators["noop"])
        msgs     = mutator.mutate(req.messages, decision)
        if _tool_compressor_on:
            msgs = _mutators["tool_result_compressor"].mutate(msgs, decision)

        response, error_type = await _forwarder.forward(decision.model, req, msgs)
        ctx.models_tried.append(decision.model.name)

        if error_type is None:
            break

        # Pinned-model failure → drop the pin and let the next attempt route freely
        if attempt == 0 and pinned_decision is not None:
            _session_pins.pop(session, None)
            ctx.session_pin = None
            pinned_decision = None
            pin_state = "evicted_post_failure"
            continue

        ctx.last_error_type = error_type
        if not _escalation.should_escalate(error_type, attempt, ctx):
            break

    latency_ms = (time.perf_counter() - t0) * 1000
    usage      = response.get("usage", {})
    cached_tokens = usage.get("_optmod_cached_tokens", 0)
    prompt_tokens_val = usage.get("prompt_tokens", 0)

    if error_type is None and decision is not None and _session_pin_enabled:
        _update_pin(session, decision.model.name, cached_tokens, prompt_tokens_val, now)

    meta = decision.meta if decision is not None else {}
    _log.append(LogEntry(
        ts=                _now_iso(),
        session_id=        session,
        task_type=         features.task_type,
        difficulty=        features.difficulty,
        token_count=       features.token_count,
        has_tools=         features.has_tools,
        language=          features.language,
        router=            _router.name,
        decision_model=    decision.model.name if decision else "unknown",
        decision_reason=   decision.reason if decision else "",
        confidence=        decision.confidence if decision else 0.0,
        mutator=           decision.mutator if decision else "noop",
        models_tried=      ctx.models_tried,
        escalation_count=  len(ctx.models_tried) - 1,
        final_model=       ctx.models_tried[-1] if ctx.models_tried else "unknown",
        ok=                error_type is None,
        error_type=        error_type,
        latency_ms=        round(latency_ms, 2),
        prompt_tokens=     prompt_tokens_val,
        completion_tokens= usage.get("completion_tokens", 0),
        cached_tokens=     cached_tokens,
        pin_state=         pin_state,
        **{k: meta[k] for k in _META_LOG_FIELDS if k in meta},
    ))

    if error_type and not response:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"optmod: all models failed ({error_type})", "type": "proxy_error"}},
        )
    if req.stream:
        return StreamingResponse(
            _completion_to_sse(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(content=response)


@app.get("/optmod/status")
async def status() -> JSONResponse:
    now = time.time()
    active_pins = sum(
        1 for p in _session_pins.values()
        if now - p.last_turn_at <= _soft_window_s
    )
    return JSONResponse(content={
        "router":              _router.name,
        "primary":             _registry.primary.name,
        "tool_compressor":     _tool_compressor_on,
        "session_pin_enabled": _session_pin_enabled,
        "active_pins":         active_pins,
        "hard_window_s":       _hard_window_s,
        "soft_window_s":       _soft_window_s,
        "laya": (
            _router.laya_status() if hasattr(_router, "laya_status") else {"mode": "off"}
        ),
        "models": [
            {"name": m.name, "tier": m.tier_name, "cost_per_1k": m.cost_per_1k}
            for m in _registry.all()
        ],
    })


@app.post("/optmod/log/clear")
async def clear_log() -> JSONResponse:
    _log.clear()
    return JSONResponse(content={"ok": True})


@app.post("/optmod/restart")
async def restart_server() -> JSONResponse:
    # Touch main.py so uvicorn --reload picks up the change and restarts the worker.
    # Without --reload this is a no-op; restart the process manually in that case.
    from optmod.stats import reload_tier_map
    reload_tier_map()
    Path("main.py").touch()
    return JSONResponse(content={"ok": True, "message": "reloading…"})


@app.post("/optmod/compressor/{state}")
async def set_compressor(state: str) -> JSONResponse:
    global _tool_compressor_on
    if state not in ("on", "off"):
        return JSONResponse(status_code=400, content={"error": f"unknown state: {state}"})
    _tool_compressor_on = state == "on"
    return JSONResponse(content={"tool_compressor": _tool_compressor_on, "ok": True})


@app.post("/optmod/laya/{mode}")
async def set_laya(mode: str) -> JSONResponse:
    if mode not in ("off", "shadow", "active"):
        return JSONResponse(status_code=400, content={"error": f"unknown mode: {mode}"})
    if not hasattr(_router, "set_laya_mode"):
        return JSONResponse(
            status_code=409,
            content={"error": f"router {_router.name} does not support laya"},
        )
    _router.set_laya_mode(mode)
    return JSONResponse(content={"laya": _router.laya_status(), "ok": True})


@app.post("/optmod/session-pin/{state}")
async def set_session_pin(state: str) -> JSONResponse:
    global _session_pin_enabled
    if state not in ("on", "off"):
        return JSONResponse(status_code=400, content={"error": f"unknown state: {state}"})
    _session_pin_enabled = state == "on"
    if not _session_pin_enabled:
        _session_pins.clear()
    return JSONResponse(content={"session_pin_enabled": _session_pin_enabled, "ok": True})


@app.post("/optmod/router/{name}")
async def set_router(name: str) -> JSONResponse:
    global _router
    valid = {"passthrough", "rule_based", "decision_tree", "trouter", "perf_router"}
    if name not in valid:
        return JSONResponse(status_code=400, content={"error": f"unknown router: {name}"})
    _router = build_router(name, _config.dict())
    return JSONResponse(content={"router": name, "ok": True})
