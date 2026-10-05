# Laya difficulty tier for PerfRouter (per-request δ and cost cap)

You are implementing this in the `optmod` repo. Read `CLAUDE.md` first: its
"Architecture Rules (Non-Negotiable)" apply to everything below. This plan was
written against `optmod` at `e55beed` and `perfrouter` at `94856e9`; if the
code has moved, trust the code and tell me what differs before changing the
design.

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | **Done** (2026-10-04) | Everything in this plan with the `embedded` backend only: Laya runs inside the optmod process. Config already carries both `embedded` and `remote` sub-blocks; `backend: remote` loads but reports `laya_status=unavailable` ("not implemented yet (phase 2)"). |
| 1b | **Done** (2026-10-04) | Low-similarity queries are classified too, observe-only (logged, never applied). See "Phase 1b" below. |
| 2 | Not started | The `remote` backend: call a `laya-serve` instance over HTTP, at localhost or any other URL. See "Phase 2: remote backend" at the end. |
| 3 | Not started, gated on 1b shadow data | Low-similarity hard rescue: when MiniLM similarity is low and Laya says `hard` confidently, route to a configured model instead of the cheapest. See "Phase 3" at the end. |

## Why

PerfRouter predicts quality per (model, task type). Every query of the same
task type therefore gets the same ranking, whether it is trivial or hard, and
one static `degradation_threshold` (δ = 0.15) and one static
`cost_cap_multiplier` (2.0) are applied to all of them.

Laya (`pip install laya`, Apache 2.0) is a small local decision model that
answers typed questions about a text in one forward pass, with no generation.
We use it to classify each routed query as easy / medium / hard, and let that
tier choose δ and the cost cap for that one request. Easy queries accept a
wider quality band (cheaper models); hard queries get a narrow band and a
higher cost cap.

The cost cap is part of this on purpose. With the current baseline
(`deepseek/deepseek-v4-flash`, $0.175 blended) and cap 2.0, only models up to
$0.35 are ever eligible. On current `perfrouter/models.yaml` pricing that
leaves six of the ten curated models; `deepseek-v4-pro`, `mimo-v2.5-pro`,
`kimi-k2.6` and `claude-sonnet-4.6` are excluded on every request. Tightening δ
alone would only reshuffle among the six cheap ones.

## Scope

In scope, all inside `optmod`:

1. A Laya tier classifier (new module) with an `embedded` (in-process) backend
   (phase 1) and a `remote` HTTP backend (phase 2).
2. Per-request δ and cost cap in `PerfRouterRouter._route`.
3. Three modes: `off`, `shadow` (compute and log, do not act), `active`.
4. Structured log fields so the effect can be measured.
5. Runtime mode toggle and status reporting.
6. Tests that need no model download and no network.

Out of scope. Do not implement these, and do not add config keys for them:

- Any change to the `perfrouter` package or to `routing/perfrouter/inference.py`.
  Both `degradation_threshold` and `cost_cap_multiplier` are already per-call
  arguments of `PerfRouterInference.route()`; that is all this feature needs.
- Using Laya to pick the task type for low-similarity queries (the
  "ambiguity rerank"). Fallback behaviour stays exactly as today.
- Fine-tuning Laya, dashboard UI changes, moving `route()` off the event loop,
  the feedback loop.

## Flow

```
chat_completions
  FeatureExtractor.extract          unchanged (regex, <1 ms, no ML)
  session-pin gate                  unchanged; hard pin skips router and Laya
  PerfRouterRouter._route
    build routing_text              unchanged
    if laya mode != off and routing_text:
        top = perf_router.classify_task(routing_text)     # MiniLM, ~2 ms
        tier = laya.classify(laya_text)                   # ~11 ms on RTX 4080
        if top[0][1] < min_similarity: tier is observe-only (phase 1b)
    resolve (delta, cost_cap) from tier, or static values
    perf_router.route(..., degradation_threshold=delta, cost_cap_multiplier=cap)
  mutate, forward, escalate         unchanged
  log                               new structured fields
```

Below `min_similarity`, `route()` takes its `fallback_ambiguous` branch and
skips XGBoost, so a tier has nothing to act on. Originally Laya was skipped
there; since phase 1b it still runs and is logged, but the static δ and cap are
used and no shadow route is made. Empty routing text skips Laya.

Calling `classify_task` in the router means MiniLM encodes the text twice (once
here, once inside `route()`). Accept that cost in this phase rather than
changing the inference class.

## New module: `routing/perfrouter/laya_tier.py`

```python
@dataclass(frozen=True)
class TierResult:
    tier:          str            # "easy" | "medium" | "hard"
    confidence:    float          # answers["tier"]["answer_confidence"]
    probabilities: dict[str, float]
    latency_ms:    float

class LayaTierClassifier:
    def __init__(self, cfg: dict) -> None: ...
    def classify(self, text: str) -> tuple[TierResult | None, str]:
        """Returns (result, status). Never raises."""
```

`status` is one of: `ok`, `error`, `timeout`, `breaker_open`, `unavailable`.

The question, as a module-level constant. Use `choice`, not `score`: ordinal
score questions are Laya's weakest primitive. Keep the labels semantic (Laya's
docs warn against `yes`/`no`/`true`/`false` as choice labels).

```python
TIER_QUESTION = {
    "tier": {
        "type": "choice",
        "instructions": "How demanding is `request` for a language model to answer well?",
        "criteria": {
            "easy":   "greeting, lookup, short factual answer, simple rewrite or formatting, one-step edit",
            "medium": "several steps, routine coding or analysis, summarising or extracting from given material",
            "hard":   "long multi-step reasoning, debugging or designing complex systems, maths or proofs, specialist knowledge",
        },
    }
}
```

State is `{"request": text}`.

Backends, selected by `backend` in config. Each has its own config sub-block
(`embedded:` / `remote:`); `merge_config()` overlays them key by key on the
defaults. An unknown backend name is status `unavailable`.

- `embedded` (phase 1): `import laya; agent = laya.load(embedded.checkpoint,
  device=embedded.device)` at init, then `agent.predict(state, TIER_QUESTION)`.
  Run one warm-up predict at init so the first real request is not slow. If
  `laya` is not installed or the load fails, log a warning once and return
  status `unavailable` forever; do not raise.
- `remote` (phase 2): see "Phase 2: remote backend". In phase 1 it is
  accepted by config and reports `unavailable`.

Parse `answers["tier"]["choice"]` and `answers["tier"]["answer_confidence"]`.
If `answer_confidence` is absent, use `max(probabilities.values())`. Anything
unparseable is status `error`.

Guards:

- **Timing.** Measure wall time for every call. In-process calls cannot be
  interrupted, so a completed call slower than `timeout_ms` still returns its
  result but counts as a slow call for the breaker.
- **Circuit breaker.** After `breaker_failures` consecutive errors, timeouts or
  slow calls, return `breaker_open` without calling Laya for
  `breaker_cooldown_s`, then try again.
- **Memo.** Cache results by SHA-1 of the text in a small LRU (512 entries), so
  escalation retries within one request do not re-run the model. Only `ok`
  results are cached.
- **One instance per process.** `POST /optmod/router/{name}` rebuilds the
  router. Keep the classifier in a module-level cache keyed by its config so a
  router swap does not reload the checkpoint.

## Changes to `routing/perfrouter/router.py`

In `__init__`, read `perf_router.laya` (see Config) and build the classifier
when mode is not `off`. Classifier construction failure must not stop the
router from loading.

In `_route`, after `user_messages` is built:

- `laya_text = "\n".join(user_messages[-3:])`, without the repeated last
  message (the repeat is a MiniLM embedding trick and wastes Laya's context).
  If longer than `max_chars`, keep the last `max_chars` characters, since the
  current message is at the end.
- Apply the flow above, then resolve the per-request parameters with a pure
  function so it can be unit-tested on its own:

```python
def resolve_tier_params(
    result: TierResult | None, status: str, mode: str, laya_cfg: dict,
    static_delta: float, static_cap: float | None,
) -> tuple[float, float | None, bool]:
    """Returns (delta, cost_cap, applied)."""
```

  Tier values apply only when `mode == "active"`, `status == "ok"` and
  `result.confidence >= min_confidence`. Otherwise return the static values and
  `applied=False`. A tier's `cost_cap: null` means no cap, exactly like the
  existing static key.

- In `shadow` mode, route with the static values. If the tier values would have
  differed, call `self._perf_router.route(...)` a second time with the tier
  values and record only its `decision_model` as `laya_shadow_model`. That
  second call skips nothing important and costs a few ms; it is what makes
  shadow data useful.

- Attach a `meta` dict to the returned `RoutingDecision` (see Logging) and add
  `tier=<tier>` and `laya=<status>` to the existing `reason` string.

- Add `set_laya_mode(mode: str) -> None` and a `laya_status() -> dict` method
  returning mode, backend, target (checkpoint for `embedded`, `base_url` for
  `remote`) and breaker state.

`route()` must still never raise. The classifier already swallows its own
errors; keep the existing outer `try/except` as the last line of defence.

### Gotcha: δ = 0 is not "best quality"

In `PerfRouterInference.route()`, band selection runs only when
`degradation_threshold > 0.0`. At exactly 0.0 it switches to the other mode,
`argmax(quality − α·cost_norm)`, which can pick a cheaper model than a narrow
band would. So the hard tier must use a small positive δ, not 0. When loading
config, log a warning for any tier whose `delta` is 0 or negative.

## Config (`config.yaml`)

Add under the existing `perf_router:` block. `Config._raw` already passes
unknown keys through, so `config.py` needs no change unless you add validation.

```yaml
perf_router:
  laya:
    mode: shadow                 # off | shadow | active
    backend: embedded            # embedded | remote (remote = phase 2)
    embedded:                    # laya runs inside the optmod process
      checkpoint: convaiinnovations/laya   # hub id or local path
      device: null               # null = let laya choose; or cuda, cuda:0, cpu
    remote:                      # laya-serve over HTTP (phase 2)
      base_url: http://127.0.0.1:8000      # localhost or any other URL
      api_key_env: LAYA_API_KEY            # optional bearer token, read from env
      model: english             # laya-serve model name: english | multilingual | typed-decisions
    timeout_ms: 80
    max_chars: 1500
    min_confidence: 0.5
    breaker_failures: 3
    breaker_cooldown_s: 60
    tiers:
      easy:   { delta: 0.25, cost_cap: 2.0 }
      medium: { delta: 0.15, cost_cap: 2.0 }
      hard:   { delta: 0.03, cost_cap: 4.0 }
```

A missing `laya:` block means `mode: off` and behaviour identical to today. All
numbers under `tiers` and `min_confidence` are starting placeholders to be
tuned from shadow data; do not treat them as validated. `hard.cost_cap: 4.0`
admits `deepseek-v4-pro` and `mimo-v2.5-pro` (both about 3.1× baseline) and
still excludes `kimi-k2.6` and `claude-sonnet-4.6`.

## Dependencies (`pyproject.toml`)

Add an optional extra, like `trouter`:

```toml
laya = ["laya>=0.3.26"]
```

`laya` 0.3.26 requires `torch>=2.0`, `transformers>=4.48`, Python >= 3.10.
Check that it resolves alongside `sentence-transformers` and the `perfrouter`
extra with `uv`, and report the result. The `remote` backend (phase 2) needs
only `httpx`, which is already a dependency; the optmod host then does not need
`laya` installed at all.

Phase 1 result: resolves cleanly with `trouter` + `perfrouter` (torch 2.12,
transformers 5.9, sentence-transformers 5.5).

## Logging

`schemas.py`:

- `RoutingDecision` gets `meta: dict = field(default_factory=dict)`.
- `LogEntry` gets these fields, all with defaults so old log lines and other
  routers keep working:

| Field | Type | Meaning |
|---|---|---|
| `pr_task_type` | `str = ""` | PerfRouter's task type (today only inside `decision_reason`) |
| `pr_routing_mode` | `str = ""` | `normal`, `fallback_ambiguous`, `fallback_empty` |
| `pr_top_similarity` | `float = 0.0` | Top MiniLM similarity |
| `laya_status` | `str = ""` | `off`, `skipped_empty`, `low_confidence`, or a classifier status |
| `laya_tier` | `str = ""` | `easy`, `medium`, `hard`, or empty |
| `laya_confidence` | `float = 0.0` | |
| `laya_ms` | `float = 0.0` | Laya wall time for this request (≈ 0 on a memo hit) |
| `laya_cached` | `bool = False` | Tier came from the classifier memo (escalation retry or repeated text), so `laya_ms` is not a model latency |
| `laya_applied` | `bool = False` | True only when tier values were used for routing |
| `laya_shadow_model` | `str = ""` | Shadow mode: model the tier values would have picked |
| `effective_delta` | `float \| None = None` | δ actually passed to `route()` |
| `effective_cost_cap` | `float \| None = None` | Cap actually passed; `None` = disabled |

`main.py`: when building the `LogEntry`, copy these from `decision.meta` if
present. Hard-pinned decisions have empty meta and get the defaults. The
existing `task_type` and `difficulty` fields stay as they are (regex features).

Do not log the query text.

## Runtime control (`main.py`)

- `POST /optmod/laya/{mode}` with mode in `off | shadow | active`, mirroring
  `/optmod/compressor/{state}`. Return 400 for an unknown mode, and 409 if the
  active router has no `set_laya_mode`.
- `GET /optmod/status` gains a `laya` key from `laya_status()`, or
  `{"mode": "off"}` when the active router is not `perf_router`.

## Tests

New `tests/test_laya_tier.py`. No test may import the real `laya` model,
download a checkpoint or open a socket. Use a fake backend object (and, in
phase 2, `respx` for the remote backend).

Classifier:

- Parses a well-formed answer into `TierResult`; falls back to max probability
  when `answer_confidence` is missing.
- Backend exception gives `(None, "error")` and does not raise.
- `backend: remote` (phase 1) and unknown backends give `unavailable`.
- Breaker opens after N consecutive failures, returns `breaker_open` without
  calling the backend, and closes after the cooldown (inject the clock).
- Memo: the same text calls the backend once.
- `laya` not installed gives `unavailable`.

`resolve_tier_params`:

- `active` + `ok` + confident uses the tier's δ and cap.
- Below `min_confidence`, non-`ok` status, or `shadow`/`off` mode returns the
  static values with `applied=False`.
- `cost_cap: null` in a tier comes through as disabled.

Router integration, with a stub inference object exposing `classify_task`,
`route` and `_model_ids` (the real artifacts and sentence-transformers are not
available in tests):

- Similarity below `min_similarity` (phase 1b): Laya is called and its tier
  logged, `laya_applied=False`, static values passed to `route()`, no shadow
  route call.
- Empty routing text: Laya is not called.
- `active`: `route()` receives the tier's `degradation_threshold` and
  `cost_cap_multiplier`.
- `shadow`: `route()` is called with static values, a second call uses tier
  values, and `laya_shadow_model` is set from it.
- `laya_text` has no repeated last message and is truncated from the left.
- With no `laya:` block, the arguments passed to `route()` are identical to
  today's.

Extend `tests/test_proxy_e2e.py` or add a small test that a decision's `meta`
reaches the JSONL line, and that a hard-pinned turn logs defaults.

The existing suite must still pass:

```bash
uv run pytest tests/test_proxy_e2e.py tests/test_routers.py tests/test_features.py \
              tests/test_escalation.py tests/test_session_pin.py \
              tests/test_tool_result_compressor.py tests/test_laya_tier.py -v
```

## Docs

Update `CLAUDE.md`: a short "Laya difficulty tier" section (modes, config
block, toggle endpoint, the δ = 0 gotcha), the new file in "Key Files", and the
new endpoint in the endpoints table.

## Done when

- With no `laya:` block or `mode: off`, routing decisions and the arguments to
  `PerfRouterInference.route()` are unchanged from today.
- In `shadow`, decisions are unchanged and every routed, above-threshold turn
  logs a tier, confidence, latency and (when it differs) a shadow model.
- In `active`, the tier's δ and cap reach `route()` only when Laya answered
  `ok` at or above `min_confidence`.
- A Laya failure, slow call or missing package never fails or delays a request
  beyond the one slow call that trips the breaker.
- All tests above pass, and `git diff` touches nothing under the `perfrouter`
  package or `routing/perfrouter/inference.py`.

When you finish, report: what you changed, the dependency resolution result,
anything in this plan that did not match the code, and anything you could not
verify (for example, a real Laya call, which needs the checkpoint and a GPU).

## What is not yet known

These are for the human running the rollout, not tasks for you:

- Whether zero-shot Laya separates easy from hard on this traffic at all. Its
  authors describe the base checkpoints as a base to fine-tune rather than a
  zero-shot decision engine, and say shipped confidences are over-confident.
  Shadow mode exists to answer this before anything acts on the tier.
- Latency on the actual host. Published figures are about 33-40 ms per call on
  a T4 and about 10 ms on a faster consumer GPU; CPU is 200 ms or more and
  should not be used in-process.
- The right δ, cap and `min_confidence` values.

## Phase 1b: classify low-similarity queries (observe-only) — done

Why: a real-GPU probe on 2026-10-04 (about 15 hand-written single-turn
prompts) showed most clearly hard prompts (a proof, a distributed rate limiter,
a lock-free Rust map, a Kubernetes OOM investigation) scoring MiniLM similarity
0.11–0.20, under `min_similarity: 0.20`. Those go to `fallback_ambiguous`, i.e.
the cheapest eligible model (a free-tier model), and before 1b Laya was never
asked about them. Lowering `min_confidence` to ~0.4 was considered and
rejected for now: on a 3-way choice the floor is 0.33, the hard verdicts seen
were 0.4959 vs 0.4288 for medium, and it would only affect queries that already
route normally.

What changed (`routing/perfrouter/router.py`): Laya runs whenever routing text
is non-empty. When top similarity is below `min_similarity`, the result is
logged (`laya_tier`, `laya_confidence`, `laya_status`) but
`resolve_tier_params` is called as if in shadow mode (static δ/cap,
`laya_applied=False`) and no shadow `route()` call is made. These rows are
identified by `pr_routing_mode == "fallback_ambiguous"`. Cost: one Laya call
(~11 ms on GPU) on requests that previously skipped it. The `skipped_low_similarity`
status no longer exists.

What to look at in the shadow log before phase 3:

- Share of routed turns with `pr_routing_mode == "fallback_ambiguous"`.
- Of those, the `laya_tier` / `laya_confidence` distribution, and a manual
  read of a sample of `hard` rows (look up the session; the query text is not
  logged) to judge whether they really are hard.
- Whether real (multi-turn, tool-heavy) agent traffic has the same low
  similarity as the single-turn probe.

## Phase 3: low-similarity hard rescue — not started

Only do this if phase 1b data shows that confident `hard` verdicts on
low-similarity turns are mostly right.

Behaviour: when `pr_routing_mode` would be `fallback_ambiguous` (top
similarity < `min_similarity`), `status == "ok"`, `tier == "hard"` and
confidence ≥ a rescue threshold, do not take the cheapest-model fallback.
Instead route to a configured rescue model. Easy/medium low-similarity turns
keep today's cheapest fallback.

Constraints: stays inside optmod. Do not change the `perfrouter` package or
`PerfRouterInference.route()` (`min_similarity_threshold` is an init
parameter, not per call, so it cannot be lowered for one request).

Sketch:

- Config, under `perf_router.laya`:
  ```yaml
  low_similarity_rescue:
    model: deepseek/deepseek-v4-flash   # baseline by default; v4-pro is the stronger option
    min_confidence: 0.5                  # separate from the normal-path threshold
  ```
  No block → rescue disabled (phase-1b behaviour).
- In `_route`, after Laya: if the rescue conditions hold and mode is `active`,
  skip `self._perf_router.route()` and build the `RoutingDecision` from the
  rescue model via `_resolve_model` (respecting the same eligibility
  `route()` enforces: vision if `has_images`, context window vs
  `token_count`; if the rescue model is ineligible, fall back to today's
  path). In `shadow`, route as today and log the rescue model as
  `laya_shadow_model`.
- Session pins: a hard pin still bypasses the router entirely; a soft pin's
  bonus does not apply to the rescue (it is an override, like vision
  escapes).
- Logging: add `laya_rescued: bool` to `LogEntry` (default False);
  `pr_routing_mode` stays `fallback_ambiguous` so rescued rows are easy to
  find; reason string gets `rescue=<model>`.
- Tests: rescue fires only for low-sim + ok + hard + confident + active;
  shadow logs the rescue model without acting; ineligible rescue model falls
  back; no `low_similarity_rescue` block behaves exactly like phase 1b.

Open questions for the human: which rescue model (baseline flash vs v4-pro,
~3.1× cost); whether `medium` low-sim turns should also leave the free tier.

## Phase 2: remote backend

Goal: run Laya as a separate `laya-serve` process (same host or another
machine with a GPU) and have optmod call it over HTTP. Useful when the optmod
host has no GPU, or so `uvicorn --reload` restarts do not reload the checkpoint.

Running the server (all config is env vars, no CLI flags):

```bash
pip install laya
LAYA_DEVICE=cuda LAYA_MODELS=english LAYA_PORT=8000 [LAYA_API_KEY=...] laya-serve
curl http://127.0.0.1:8000/health      # {"status": "ok", "loaded": [...], "device": ...}
```

Config is already in place from phase 1; switching is `backend: remote`:

```yaml
perf_router:
  laya:
    backend: remote
    remote:
      base_url: http://gpu-box:8000   # or http://127.0.0.1:8000
      api_key_env: LAYA_API_KEY       # put the key in optmod's .env
      model: english                  # laya-serve name, NOT a hub id
```

Implementation, in `routing/perfrouter/laya_tier.py`:

- Add `_RemoteBackend` and select it for `backend == "remote"` in
  `LayaTierClassifier.__init__` (replacing the phase-1 `_Unavailable` stub).
- One `httpx.Client` created at init (never per request) with
  `timeout = timeout_ms / 1000`. `POST {base_url}/v1/systemone` with body
  `{"state": ..., "questions": ..., "model": remote.model}`. If the env var
  named by `api_key_env` is set, send `Authorization: Bearer <key>`. The
  response has the same `answers` shape as `agent.predict()`.
- `laya-serve` resolves short model names (`english`, `multilingual`,
  `typed-decisions`), not hub ids — hence the separate `remote.model` key.
- Map `httpx.TimeoutException` to status `timeout` (the phase-1 code catches
  `TimeoutError` only). Non-2xx (`raise_for_status`) and connection errors are
  `error`. Both count toward the breaker as today.
- Optional: probe `GET /health` at init and log the loaded models/device;
  failure is a warning, not `unavailable` (the server may come up later).
- Note: `route()` runs on the event loop and `httpx.Client` is sync, so a
  remote call blocks the loop for up to `timeout_ms`, like the embedded call.
  Moving `route()` off the loop stays out of scope.

Reference implementation (written during phase 1, then removed to keep phase 1
embedded-only; adapt `cfg[...]` to the `remote:` sub-block):

```python
class _RemoteBackend:
    def __init__(self, cfg: dict) -> None:
        rem = cfg["remote"]
        headers = {}
        key_env = rem.get("api_key_env")
        key = os.environ.get(key_env, "") if key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._url = rem["base_url"].rstrip("/") + "/v1/systemone"
        self._model = rem.get("model") or None
        self._client = httpx.Client(timeout=float(cfg["timeout_ms"]) / 1000.0, headers=headers)

    def predict(self, state: dict, questions: dict) -> dict:
        body: dict = {"state": state, "questions": questions}
        if self._model:
            body["model"] = self._model
        resp = self._client.post(self._url, json=body)
        resp.raise_for_status()
        return resp.json()
```

Phase 2 tests (`respx`, no real socket):

- Request shape: URL, JSON body incl. `model`, bearer header when the env var
  is set and absent when not.
- `httpx.ReadTimeout` gives `(None, "timeout")`; 5xx gives `(None, "error")`;
  both trip the breaker after `breaker_failures`.
- Replace `test_remote_backend_is_unavailable_in_phase_1` with a happy-path
  test; `status()["target"]` is the `base_url`.

Phase 2 docs: update the "Laya Difficulty Tier" section in `CLAUDE.md`
(backends line and the run instructions above).
