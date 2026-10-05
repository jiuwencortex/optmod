# optmod — Claude Code Context

This file is read automatically at the start of every Claude Code session.

## What This Project Is

`optmod` is a **local OpenAI-compatible routing proxy** (FastAPI, Python 3.11+).
Any agent pointing at `http://localhost:8765/v1` gets transparent multi-model routing:
classify → route → mutate context → forward → escalate on error → log → respond.

The calling agent never knows a proxy is in the middle.

## Running the Server

```bash
uv run --env-file .env uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```

API keys are in `.env` and must be passed via `--env-file .env` (uv does not auto-load `.env`).
They are resolved at startup via `api_key_env` in `config.yaml`.
**Important:** uvicorn `--reload` only watches `.py` files. After changing `config.yaml`,
do a hard restart (Ctrl+C → re-run).

Dashboard: `http://localhost:8765/ui`

## Current Model Pool (multi-provider — DeepSeek direct + OpenRouter)

Primary model: `deepseek-v4-flash` (set via `primary_model` in `config.yaml`).

| Tier | Model | Provider | Cost/1k in | Notes |
|------|-------|----------|-----------|-------|
| fast (0) | `openai/gpt-oss-120b:free` | OpenRouter | $0.0 | Free tier |
| fast (0) | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter | $0.0 | Free tier, `thinking_mode=true` |
| fast (0) | `openai/gpt-4o-mini` | OpenRouter | $0.00015 | |
| reasoning (1) | `deepseek-v4-flash` | DeepSeek direct | $0.00014 | Primary |
| reasoning (1) | `google/gemini-3.1-flash-lite` | OpenRouter | $0.0001 | |
| reasoning (1) | `tencent/hy3-preview` | OpenRouter | $0.000063 | `thinking_mode=true` |
| oracle (2) | `deepseek-v4-pro` | DeepSeek direct | $0.000435 | `thinking_mode=true` |
| oracle (2) | `anthropic/claude-sonnet-4.6` | OpenRouter | $0.003 | `thinking_mode=true` |
| oracle (2) | `xiaomi/mimo-v2.5-pro` | OpenRouter | $0.000435 | |
| oracle (2) | `moonshotai/kimi-k2.6` | OpenRouter | $0.000684 | |

Escalation path: `fast → reasoning → oracle → 502`. Required env vars: `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` (both in `.env`).

## Active Routers (swap with `POST /optmod/router/{name}`)

| Name | Description |
|------|-------------|
| `perf_router` | **Default.** Sentence-BERT (`all-MiniLM-L6-v2`) classifies the query into task types, XGBoost predicts per-model quality, then `adjusted_utility = quality − α·cost_norm` picks the winner. Knobs: `perf_router_cost_weight` (α), `perf_router_baseline`, `perf_router_degradation_threshold`, `perf_router_min_similarity`. Loads `routing/perf_router.pkl` + `task_taxonomy.json` + `model_registry.json` + `model_features.csv`. |
| `rule_based` | 9 deterministic rules (task type, difficulty, language, tokens) |
| `trouter` | Neural net: sentence-BERT → MLP → model index. Loads `routing/trouter_weights.pt` |
| `passthrough` | Always uses the primary model. Useful as a baseline |
| `decision_tree` | scikit-learn tree; falls back to `rule_based` if `routing_policy.pkl` absent |

## Session Pinning

Multi-turn conversations stick to the same model while the provider-side prompt cache is warm. Session ID = SHA1 of the first user message.

- **0–300 s** since last turn → **hard pin**: bypass the router entirely, force the previous model.
- **300–1800 s** → **soft bonus**: `perf_router` scores normally, but the previously-used model is credited with its observed cache savings (`pin.last_cache_rate × cost_norm[pin_idx] × session_pin_soft_bonus_weight`). Applied in all three selection branches (standard, degradation, fallback).
- **> 1800 s** → fresh routing.

Escapes (always override the pin): pinned model error/rate-limit (drop pin, re-route on next attempt); has-images request against a non-vision pin; pinned model removed from registry; context overflow → **rehome** to the closest larger-context model (same tier preferred, tiebreak by `|Δcost|`); no candidate fits → drop pin.

Pin state is in-memory only (`_session_pins: dict[str, SessionPin]` in `main.py`), evicted lazily on access. Wiped on restart. Each `LogEntry` carries `pin_state` ∈ {`fresh`, `hard`, `soft`, `rehomed_context`, `evicted_vision`, `evicted_missing_model`, `evicted_context`, `evicted_post_failure`} and `cached_tokens` (extracted defensively from `usage.prompt_tokens_details.cached_tokens`, `prompt_cache_hit_tokens`, or `cache_read_input_tokens`).

Config knobs: `session_pin_hard_window_s`, `session_pin_soft_window_s`, `session_pin_soft_bonus_weight`. Per-model `supports_vision: true` declares vision capability.

## Laya Difficulty Tier (perf_router only)

[Laya](https://github.com/nandhakishorm/laya) (optional extra: `uv pip install -e '.[laya]'`) classifies each
routed query as `easy` / `medium` / `hard` with one typed `choice` question, and the tier picks the
per-request δ (`degradation_threshold`) and `cost_cap_multiplier` passed to `PerfRouterInference.route()`.

- **Modes** (`perf_router.laya.mode`): `off` (identical to no block), `shadow` (route with static values;
  if tier values differ, a second `route()` call records `laya_shadow_model`), `active` (tier values used
  only when Laya answered `ok` with confidence ≥ `min_confidence`).
- Laya runs after the hard-pin gate on any non-empty routing text. Below MiniLM `min_similarity` it is
  **observe-only**: `route()` takes `fallback_ambiguous` (cheapest model, δ ignored), so the tier is logged
  but never applied (rows have `pr_routing_mode=fallback_ambiguous`). Acting on it is phase 3 in the plan.
- Input = last 3 user messages **without** the repeated last one, left-truncated to `max_chars`.
- Backends (`backend:`), each with its own sub-block: `embedded` (phase 1, implemented) runs `laya.load`
  in-process with a warm-up predict at init; `remote` (phase 2, **not implemented** — reports
  `unavailable`) will call `laya-serve` at `remote.base_url` (localhost or any URL). Missing package or
  load failure → status `unavailable`, never an error. Plan: `optmod-spec/plans/laya-difficulty-tier.md`.
- Guards: wall-time per call; `breaker_failures` consecutive errors/timeouts/slow calls (> `timeout_ms`)
  open a breaker for `breaker_cooldown_s`; 512-entry SHA-1 memo of `ok` results; one classifier per
  process per config (router swaps don't reload the checkpoint).
- Toggle: `POST /optmod/laya/{off|shadow|active}`. Status: `laya` key in `GET /optmod/status`.
- **δ = 0 is not "best quality"**: in `route()`, δ = 0 switches to `argmax(quality − α·cost_norm)`.
  The hard tier must use a small positive δ; a warning is logged for any tier with δ ≤ 0.

```yaml
perf_router:
  laya:
    mode: shadow                 # off | shadow | active
    backend: embedded            # embedded | remote (remote = phase 2)
    embedded:
      checkpoint: convaiinnovations/laya
      device: null               # or cuda, cuda:0, cpu
    remote:                      # phase 2
      base_url: http://127.0.0.1:8000
      api_key_env: LAYA_API_KEY
      model: english             # laya-serve name, not a hub id
    timeout_ms: 80
    max_chars: 1500
    min_confidence: 0.5
    breaker_failures: 3
    breaker_cooldown_s: 60
    tiers:                       # placeholders — tune from shadow data
      easy:   { delta: 0.25, cost_cap: 2.0 }
      medium: { delta: 0.15, cost_cap: 2.0 }
      hard:   { delta: 0.03, cost_cap: 4.0 }   # cost_cap: null = no cap
```

Each `LogEntry` carries `pr_task_type`, `pr_routing_mode`, `pr_top_similarity`, `laya_status`
(`off`, `skipped_empty`, `low_confidence`, or a classifier status: `ok`,
`error`, `timeout`, `breaker_open`, `unavailable`), `laya_tier`, `laya_confidence`, `laya_ms`,
`laya_applied`, `laya_shadow_model`, `effective_delta`, `effective_cost_cap` — copied from
`RoutingDecision.meta`. Hard-pinned turns log defaults. Query text is never logged.

## Mutators

Post-routing context transforms. `BaseContextMutator.mutate(messages, decision) → new list` — never raise, never modify in-place.

| Name | Purpose |
|------|---------|
| `noop` | Identity passthrough |
| `thinking_mode` | Prepends `/think` to trigger CoT in thinking models. Auto-selected when `decision.model.thinking_mode == True` |
| `tool_result_compressor` | RTK-style compression of tool-role messages (git diff/status, grep, ls/tree, build/test output, log dedup, smart-truncate). Runtime toggle via `POST /optmod/compressor/{on|off}` |

## Key Files

```
main.py                              FastAPI app, lifespan, all endpoints, session-pin gate + helpers
config.yaml                          Model pool, active router, perf_router knobs, session_pin_* knobs
config.py                            Config loader — reads config.yaml + resolves env vars; _raw passes
                                     all custom keys through to routers
registry.py                          ModelConfig (incl. supports_vision), ModelRegistry
schemas.py                           Pydantic + dataclass types: OpenAIChatRequest, ChatMessage, Features,
                                     RoutingContext (session_pin field), RoutingDecision, LogEntry
                                     (cached_tokens + pin_state), SessionPin
features.py                          FeatureExtractor — <1ms regex classifier, no I/O
forwarder.py                         httpx async forwarder, one AsyncClient per base_url; defensive
                                     cached_tokens extraction into usage._optmod_cached_tokens
escalation.py                        EscalationPolicy
log.py                               RoutingLog — append-only JSONL, threading.Lock
stats.py                             /api/stats aggregation incl. cache_hit_rate, cached_tokens
                                     totals; _TIER_MAP loaded from config.yaml at import
routing/__init__.py                  BaseRouter ABC + build_router() factory
routing/passthrough.py               PassthroughRouter — primary model
routing/rule_based.py                9-rule deterministic router
routing/decision_tree.py             scikit-learn tree (falls back to rule_based if pkl absent)
routing/trouter_router.py            TRouter — sentence-BERT + MLP
routing/train_trouter.py             TRouter training code + standalone route() function
routing/perf_router_router.py        PerfRouter config plumbing; resolves session_pin → inference id
routing/perf_router_inference.py     PerfRouter inference (sentence-BERT + XGBoost); soft-pin bonus
                                     applied in all three selection branches
routing/perfrouter/laya_tier.py      LayaTierClassifier (embedded backend; remote = phase 2), breaker, memo, TIER_QUESTION
routing/perf_router.pkl              Trained PerfRouter checkpoint
routing/task_taxonomy.json           Task type taxonomy for sentence-BERT classification
routing/model_registry.json          Per-model metadata (effective context, vision capability)
routing/model_features.csv           Per-model price + capability features
mutators/noop.py                     Identity
mutators/thinking_mode.py            Prepends /think for CoT-capable models
mutators/tool_result_compressor.py   RTK-style compression of tool-role messages
ui/index.html                        Self-contained dashboard (no build step)
tests/test_proxy_e2e.py              Mock e2e via respx
tests/test_routers.py                Unit tests — all routers
tests/test_features.py               Unit tests — FeatureExtractor
tests/test_escalation.py             Unit tests — EscalationPolicy
tests/test_session_pin.py            Unit tests — pin helpers, escapes, rehoming, cache extraction
tests/test_tool_result_compressor.py Unit tests — every compression filter and edge case
tests/test_laya_tier.py              Unit tests — Laya classifier, resolve_tier_params, router + log plumbing
tests/test_live_e2e.py               Live tests hitting real providers — skipped if no API key
```

## Architecture Rules (Non-Negotiable)

- **Never raise** inside `router.route()` — catch everything, return passthrough
- **Never mutate messages in-place** — mutators return a new list
- **Feature extraction must be <1ms** — no ML, no I/O inside `FeatureExtractor.extract()`
- **All regex compiled at module import time**, not inside functions
- **One `httpx.AsyncClient` per `base_url`** — never create per request
- **JSONL log**: one line per request, appended atomically — never rewrite
- **`pyproject.toml`** is the only dependency file — no `requirements.txt`
- **`asyncio_mode = "auto"`** in pytest config

## Testing

```bash
# Full mock suite (fast, no network)
uv run pytest tests/test_proxy_e2e.py tests/test_routers.py \
              tests/test_features.py tests/test_escalation.py \
              tests/test_session_pin.py tests/test_tool_result_compressor.py \
              tests/test_laya_tier.py -v

# Live e2e (requires OPENROUTER_API_KEY + DEEPSEEK_API_KEY in .env)
uv run pytest tests/test_live_e2e.py -v -s
```

Mock tests use `respx` to intercept `httpx` calls. Note that the current
config mixes providers: `test_proxy_e2e.py` would need updating to mock
both `https://openrouter.ai/api/v1` and `https://api.deepseek.com` if you
exercise paths that touch DeepSeek direct.

## Dashboard Endpoints

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/optmod/log/clear` | Truncates `routing.log.jsonl` |
| `POST` | `/optmod/restart` | Re-reads `_TIER_MAP` + touches `main.py` for `--reload` |
| `POST` | `/optmod/router/{name}` | Hot-swap router (`perf_router`, `rule_based`, `trouter`, `passthrough`, `decision_tree`) |
| `POST` | `/optmod/compressor/{state}` | Toggle the `tool_result_compressor` mutator (`on` / `off`) |
| `POST` | `/optmod/laya/{mode}` | Laya tier mode (`off` / `shadow` / `active`); 409 if active router isn't `perf_router` |
| `GET` | `/optmod/status` | Current router, primary, model list with costs, `active_pins`, `hard_window_s`, `soft_window_s`, `tool_compressor`, `laya` |
| `GET` | `/api/stats?range=N` | Aggregated stats (1h, 6h, 24h, last-N, all) — includes `cache_hit_rate`, `cached_tokens`, per-model `model_tokens.{prompt,completion,cached}` |
| `GET` | `/api/stats/live` | Lightweight live counts for polling |
| `GET` | `/ui` | Live dashboard (5s auto-refresh, includes cache-hit-rate tile + pinned-session counter) |

## Common Gotchas

- **`config.yaml` not picked up after edit** — uvicorn `--reload` ignores `.yaml`.
  Hard-restart the server, or call `POST /optmod/restart` (re-reads `_TIER_MAP` in-process).
- **All tier badges show ORACLE** — `_TIER_MAP` in `stats.py` not loaded with current config.
  Hard-restart the server.
- **Thinking models need large `max_tokens`** — models with `thinking_mode: true`
  in `config.yaml` consume token budget on internal reasoning before producing
  the answer. Use `max_tokens >= 1500` in tests.
- **Old log entries from a previous model pool** appear in the dashboard.
  Use the **CLEAR LOG** button or `POST /optmod/log/clear`.
- **TRouter / PerfRouter optional deps** — both import `torch` /
  `sentence_transformers`; PerfRouter additionally needs `xgboost`. Install with
  `uv pip install -e '.[trouter]'`. If absent, the router falls back silently.
- **`Config.dict()` and custom keys** — `Config._raw` spreads all top-level
  YAML keys through `Config.dict()`, so any new `config.yaml` key is
  automatically available to routers/mutators without modifying `config.py`.
  Don't add hardcoded fields to `Config.dict()` unless they need defaults.
- **`stream_options` is excluded from forwarded payload** — DeepSeek direct API
  400s if it sees `stream_options` while `stream=False`. The forwarder excludes
  it; don't re-add it without testing both providers.
- **Session pin survives router swap** — `POST /optmod/router/{name}` doesn't
  clear `_session_pins`. The pin (and hard-pin behaviour) is router-agnostic;
  the soft bonus is perf_router-only.

## What's Not Yet Built

- `routing_policy.pkl` — WildClawBench-trained scikit-learn tree; needed to enable `decision_tree` router without falling back to `rule_based`
- Hermes plugin — thin wrapper calling `/optmod/*` control endpoints
- Feedback loop CLI — retrains TRouter / decision tree / PerfRouter from `routing.log.jsonl`
- File-backed session-pin store — current implementation is in-memory only and lost on restart
