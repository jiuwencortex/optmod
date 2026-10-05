# optmod

A local OpenAI-compatible routing proxy that selects the optimal LLM for each request — transparently, within a single HTTP call.

<img width="1834" height="852" alt="image" src="https://github.com/user-attachments/assets/1ee67d2c-4a20-43ab-b3cc-4c13ad62f555" />

Point any OpenAI-compatible agent (Hermes, LangChain, etc.) at `http://localhost:8765/v1` and optmod classifies each request, picks the right model, escalates on failure, and returns one clean response. The calling agent never knows a proxy is in the middle.

## How it works

```
Agent  →  POST /v1/chat/completions  →  optmod
                                           │
                                           ├─ extract features  (<1ms, regex only)
                                           ├─ route → pick model + strategy
                                           ├─ mutate context (e.g. /think prefix)
                                           ├─ forward via OpenRouter (httpx async)
                                           │     └─ on error: escalate up one tier
                                           └─ log to JSONL → return response
```

## Model pool

The current pool is multi-provider: DeepSeek direct API for the flagship models, [OpenRouter](https://openrouter.ai) for everything else. Set both `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY` in `.env`; each model declares which env var to use via `api_key_env` in `config.yaml`.

| Tier | Model | Provider | Cost / 1k input |
|---|---|---|---|
| fast (0) | `openai/gpt-oss-120b` | OpenRouter | $0.000037 |
| fast (0) | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter | $0.0 |
| fast (0) | `openai/gpt-4o-mini` | OpenRouter | $0.00015 |
| reasoning (1) | `deepseek-v4-flash` | DeepSeek | $0.00014 |
| reasoning (1) | `google/gemini-3.1-flash-lite` | OpenRouter | $0.0001 |
| reasoning (1) | `tencent/hy3-preview` | OpenRouter | $0.000063 |
| oracle (2) | `deepseek-v4-pro` | DeepSeek | $0.000435 |
| oracle (2) | `anthropic/claude-sonnet-4.6` | OpenRouter | $0.003 |
| oracle (2) | `xiaomi/mimo-v2.5-pro` | OpenRouter | $0.000435 |
| oracle (2) | `moonshotai/kimi-k2.6` | OpenRouter | $0.000684 |

Escalation path: `fast → reasoning → oracle → 502`. Edit `config.yaml` and hard-restart (or `POST /optmod/restart`) to change the pool.

## Quickstart

```bash
# 1. Install
uv venv && uv pip install -e '.[dev]'

# 2. Add provider API keys (both required by the default config)
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
echo "DEEPSEEK_API_KEY=sk-..."      >> .env

# 3. Run — `uv run` must be told to load .env (it doesn't by default)
uv run --env-file .env uvicorn main:app --host 0.0.0.0 --port 8765 --reload

# 4. Verify
curl http://localhost:8765/optmod/status

# 5. Open the dashboard
open http://localhost:8765/ui

# 6. Send a request
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"optmod","messages":[{"role":"user","content":"why does quicksort fail on sorted input?"}]}'

# 7. Swap router at runtime (no restart needed)
curl -X POST http://localhost:8765/optmod/router/rule_based
curl -X POST http://localhost:8765/optmod/router/trouter
curl -X POST http://localhost:8765/optmod/router/passthrough
```

### TRouter — neural net routing (optional)

```bash
# Install PyTorch + sentence-transformers extras
uv pip install -e '.[dev,trouter]'

# Activate (loads trouter_weights.pt + all-MiniLM-L6-v2 encoder at startup)
curl -X POST http://localhost:8765/optmod/router/trouter
```

## Routers

| Name | Description |
|---|---|
| `passthrough` | Always routes to the primary model; no classification |
| `rule_based` | 9 deterministic rules on task type, difficulty, language, token count |
| `decision_tree` | scikit-learn tree trained on WildClawBench; falls back to `rule_based` if `routing_policy.pkl` is absent |
| `trouter` | Neural network router: sentence-BERT encodes the query, a lightweight MLP picks the model weighting quality and cost |
| `perf_router` | **Default.** Sentence-BERT classifies the query into task types, XGBoost predicts per-model quality, then `quality − α·cost` picks the winner. Tunable via `perf_router_cost_weight` (α), `perf_router_baseline`, `perf_router_degradation_threshold`, `perf_router_min_similarity` in `config.yaml`. |

All routers swap at runtime — `POST /optmod/router/{name}`, no restart required.

## Session pinning

Multi-turn conversations stick to the same model while the provider-side prompt cache (DeepSeek auto-cache, Anthropic explicit cache) is still warm — preserving cached-read pricing rather than paying full input cost on every turn.

Three phases keyed on time since the last turn in the session:

| Phase | Window | Behaviour |
|---|---|---|
| Hard pin | 0 – 300 s | Bypass the router entirely; force the previous model |
| Soft bonus | 300 – 1800 s | Router scores normally, but the previously-used model is credited with its observed cache savings |
| Fresh | > 1800 s | Pin dropped, normal routing |

Escapes that always override the pin:

- Pinned model errors / rate-limits — drop pin, fall through to the router on the next attempt
- Request includes images and the pinned model isn't vision-capable
- Pinned model removed from the registry
- Conversation grew past the pinned model's `context_window` → **rehome** to the closest larger-context model (same tier preferred, tiebreak by cost); if no candidate fits, drop the pin

Session identity is the SHA1 of the first user message in the request. Pin state is in-memory only; restarting the server clears it.

Config knobs in `config.yaml`:

```yaml
session_pin_hard_window_s: 300       # 5 min — provider cache TTL
session_pin_soft_window_s: 1800      # 30 min — soft-bonus tail
session_pin_soft_bonus_weight: 0.5   # 0.0 disables the soft bonus
```

Per-model `supports_vision: true` declares vision capability for the escape clause.

Each log entry carries a `pin_state` field — one of `fresh`, `hard`, `soft`, `rehomed_context`, `evicted_vision`, `evicted_missing_model`, `evicted_context`, `evicted_post_failure` — so you can audit what the pin did on every request.

## Dashboard

Open `http://localhost:8765/ui` for a live dashboard with 5s auto-refresh:

- **Stat cards** — total requests, success rate, escalation rate, error rate, cache hit rate (with active session-pin count)
- **Model distribution** — tier-colored usage bars, per-model cost estimate for the window, savings vs. always routing to oracle
- **Task types** — distribution of task classifications
- **Latency histogram** — bucketed with P50/P90/P99/avg
- **Escalation flow** — which models are being escalated to and why
- **Recent requests table** — per-request model (with full name tooltip), models-tried chain, cost, latency, status, routing reason (truncated; hover for full text), confidence. New rows slide in animated; a pause/resume button freezes the table without stopping stat updates.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main proxy — OpenAI wire format |
| `GET` | `/optmod/status` | Router name, primary model, full model list with costs, active session-pin count, pin window config, tool-compressor state |
| `POST` | `/optmod/router/{name}` | Swap router: `passthrough`, `rule_based`, `decision_tree`, `trouter`, `perf_router` |
| `POST` | `/optmod/compressor/{state}` | Toggle the `tool_result_compressor` mutator (`on` / `off`) |
| `POST` | `/optmod/log/clear` | Truncate `routing.log.jsonl` |
| `POST` | `/optmod/restart` | Re-read `config.yaml` tier map and touch `main.py` so `uvicorn --reload` picks up changes |
| `GET` | `/api/stats?range=N` | Aggregated stats from JSONL log (1h, 6h, 24h, last-N, all) |
| `GET` | `/api/stats/live` | Lightweight live counts for polling |
| `GET` | `/ui` | Live routing dashboard |

## Project layout

Flat layout — source lives at the project root, importable as `optmod.*`.

```
main.py               FastAPI app, lifespan, proxy endpoint
schemas.py            Pydantic + dataclass types
config.py             Config loader (config.yaml)
config.yaml           Model pool + router config (all via OpenRouter)
registry.py           ModelConfig + ModelRegistry
features.py           FeatureExtractor (<1ms regex classifier)
forwarder.py          Async httpx forwarder, one client per base_url
escalation.py         EscalationPolicy
log.py                Append-only JSONL log
stats.py              /api/stats aggregation (model_tokens, cost tracking)
routing/
  __init__.py              BaseRouter ABC + build_router() factory
  passthrough.py           PassthroughRouter
  rule_based.py            RuleBasedRouter (9 rules)
  decision_tree.py         DecisionTreeRouter (scikit-learn)
  trouter/                 TRouter package
    router.py              TRouterRouter (sentence-BERT + MLP)
    trouter_weights.pt     Trained TRouter checkpoint
    train_trouter.py       TRouter training code
  perf_router_router.py    PerfRouterRouter — config plumbing, session-pin lookup
  perf_router_inference.py PerfRouter inference engine (sentence-BERT + XGBoost)
  perf_router.pkl          Trained PerfRouter checkpoint
  task_taxonomy.json       Task type taxonomy for sentence-BERT classification
  model_registry.json      Per-model metadata (effective context, vision capability)
  model_features.csv       Per-model price + capability features
mutators/
  noop.py                  NoopMutator — passthrough
  thinking_mode.py         ThinkingModeMutator — prepends /think for CoT-capable models
  tool_result_compressor.py ToolResultCompressorMutator — RTK-style compression of tool outputs (git diff/status, grep, ls/tree, build/test output, log dedup, smart-truncate). Runtime toggle via POST /optmod/compressor/{on|off}.
tests/
  test_features.py              Unit tests — FeatureExtractor
  test_routers.py               Unit tests — all routers
  test_escalation.py            Unit tests — EscalationPolicy
  test_proxy_e2e.py             Mock e2e tests (respx, no network)
  test_session_pin.py           Unit tests — session pin helpers, escapes, rehoming, cache-token extraction
  test_tool_result_compressor.py Unit tests — every compression filter and edge case
  test_live_e2e.py              Live e2e tests (real provider calls, skipped without API keys)
ui/index.html         Self-contained stats dashboard
.env                  API keys — git-ignored
```

## Running tests

```bash
# Mock tests only (no network, fast)
uv run pytest tests/test_proxy_e2e.py tests/test_routers.py \
              tests/test_features.py tests/test_escalation.py -v

# All tests including live e2e (requires OPENROUTER_API_KEY in .env)
uv run pytest tests/ -v -s
```

## Use with Hermes

Add to `~/.hermes/config.yaml`:

```yaml
provider: custom
model: optmod-router
base_url: http://localhost:8765/v1
api_key: optmod
```

## What's not built yet

- `routing_policy.pkl` — produced by running WildClawBench; needed to enable the `decision_tree` router without falling back to `rule_based`
- Hermes plugin (thin wrapper calling `/optmod/*` endpoints)
- Feedback loop CLI (retrains TRouter / decision tree / PerfRouter from `routing.log.jsonl`)
- File-backed session-pin store (current implementation is in-memory only and lost on restart)
