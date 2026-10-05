"""
Laya difficulty tier classifier for PerfRouter.

Asks Laya one typed `choice` question ("how demanding is this request?") and
returns easy / medium / hard. PerfRouterRouter uses the tier to pick a
per-request degradation threshold (δ) and cost cap.

classify() never raises. Failures come back as a status string:
  ok | error | timeout | breaker_open | unavailable

Backends (perf_router.laya.backend):
  embedded — laya runs inside this process (phase 1)
  remote   — laya-serve over HTTP, localhost or another host (phase 2; not
             implemented yet, reports `unavailable`)
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

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

TIERS = ("easy", "medium", "hard")

_MEMO_SIZE = 512

BACKENDS = ("embedded", "remote")

DEFAULTS: dict = {
    "mode":               "off",
    "backend":            "embedded",
    "embedded": {
        "checkpoint":     "convaiinnovations/laya",
        "device":         None,
    },
    "remote": {
        "base_url":       "http://127.0.0.1:8000",
        "api_key_env":    "LAYA_API_KEY",
        "model":          "english",
    },
    "timeout_ms":         80,
    "max_chars":          1500,
    "min_confidence":     0.5,
    "breaker_failures":   3,
    "breaker_cooldown_s": 60,
    "tiers":              {},
}


def merge_config(cfg: dict | None) -> dict:
    """DEFAULTS overlaid with cfg; the embedded/remote sub-blocks merge key by key."""
    out = copy.deepcopy(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k in ("embedded", "remote") and isinstance(v, dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class TierResult:
    tier:          str            # "easy" | "medium" | "hard"
    confidence:    float          # answers["tier"]["answer_confidence"]
    probabilities: dict[str, float]
    latency_ms:    float


class _Unavailable(Exception):
    """Backend cannot run at all (package missing, load failed)."""


class _EmbeddedBackend:
    def __init__(self, cfg: dict) -> None:
        try:
            import laya  # noqa: PLC0415 — optional dependency
        except ImportError as exc:
            raise _Unavailable(f"laya not installed: {exc}") from exc
        emb = cfg["embedded"]
        kwargs = {"device": emb["device"]} if emb.get("device") else {}
        try:
            self._agent = laya.load(emb["checkpoint"], **kwargs)
            # Warm-up so the first real request does not pay for lazy init.
            self._agent.predict({"request": "hello"}, TIER_QUESTION)
        except Exception as exc:
            raise _Unavailable(f"laya load failed: {exc}") from exc

    def predict(self, state: dict, questions: dict) -> dict:
        return self._agent.predict(state, questions)


def _parse(raw: dict) -> tuple[str, float, dict[str, float]]:
    """Extract (tier, confidence, probabilities). Raises on anything unexpected."""
    ans = raw["answers"]["tier"]
    tier = ans["choice"]
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
    conf = ans.get("answer_confidence")
    if conf is None:
        if not probs:
            raise ValueError("no answer_confidence and no probabilities")
        conf = max(probs.values())
    return tier, float(conf), probs


class LayaTierClassifier:
    def __init__(
        self,
        cfg: dict,
        backend: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = merge_config(cfg)
        self._clock = clock
        self._timeout_ms = float(self._cfg["timeout_ms"])
        self._breaker_failures = int(self._cfg["breaker_failures"])
        self._breaker_cooldown_s = float(self._cfg["breaker_cooldown_s"])

        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until: float | None = None
        self._memo: OrderedDict[str, TierResult] = OrderedDict()

        self._backend = backend
        self._unavailable_reason: str | None = None
        if self._backend is None:
            try:
                kind = self._cfg["backend"]
                if kind == "embedded":
                    self._backend = _EmbeddedBackend(self._cfg)
                elif kind == "remote":
                    raise _Unavailable("remote backend not implemented yet (phase 2)")
                else:
                    raise _Unavailable(f"unknown backend {kind!r}; expected one of {BACKENDS}")
            except _Unavailable as exc:
                self._unavailable_reason = str(exc)
                logging.warning("[optmod] Laya tier classifier unavailable: %s", exc)
            except Exception as exc:
                self._unavailable_reason = str(exc)
                logging.warning("[optmod] Laya tier classifier failed to init: %s", exc)

    # ── public ────────────────────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[TierResult | None, str]:
        """Returns (result, status). Never raises."""
        try:
            return self._classify(text)
        except Exception as exc:  # last line of defence
            logging.warning("[optmod] Laya classify error: %s", exc)
            return None, "error"

    def status(self) -> dict:
        now = self._clock()
        with self._lock:
            if self._backend is None:
                breaker = "unavailable"
            elif self._open_until is not None and now < self._open_until:
                breaker = "open"
            else:
                breaker = "closed"
            return {
                "backend":              self._cfg["backend"],
                "target":               target(self._cfg),
                "breaker":              breaker,
                "consecutive_failures": self._consecutive_failures,
                "unavailable_reason":   self._unavailable_reason,
            }

    # ── internals ─────────────────────────────────────────────────────────────

    def _classify(self, text: str) -> tuple[TierResult | None, str]:
        if self._backend is None:
            return None, "unavailable"

        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None:
                self._memo.move_to_end(key)
                return hit, "ok"
            if self._open_until is not None:
                if self._clock() < self._open_until:
                    return None, "breaker_open"
                # Cooldown elapsed: half-open, allow one try.
                self._open_until = None

        t0 = time.perf_counter()
        status = "ok"
        result: TierResult | None = None
        try:
            raw = self._backend.predict({"request": text}, TIER_QUESTION)
            latency_ms = (time.perf_counter() - t0) * 1000
            tier, conf, probs = _parse(raw)
            result = TierResult(tier=tier, confidence=conf,
                                probabilities=probs, latency_ms=round(latency_ms, 2))
        except TimeoutError:
            status = "timeout"
        except Exception as exc:
            logging.debug("[optmod] Laya backend error: %s", exc)
            status = "error"
        latency_ms = (time.perf_counter() - t0) * 1000

        # A completed call slower than the budget still returns its result
        # (in-process calls cannot be interrupted) but counts against the breaker.
        failed = status != "ok" or latency_ms > self._timeout_ms
        with self._lock:
            if failed:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._breaker_failures:
                    self._open_until = self._clock() + self._breaker_cooldown_s
                    self._consecutive_failures = 0
                    logging.warning(
                        "[optmod] Laya breaker open for %.0fs (last status=%s, %.1f ms)",
                        self._breaker_cooldown_s, status, latency_ms,
                    )
            else:
                self._consecutive_failures = 0
            if result is not None:
                self._memo[key] = result
                if len(self._memo) > _MEMO_SIZE:
                    self._memo.popitem(last=False)
        return result, status


def target(cfg: dict) -> str:
    """Human-readable 'what are we talking to' for status output."""
    if cfg.get("backend") == "remote":
        return cfg["remote"]["base_url"]
    return cfg["embedded"]["checkpoint"]


# ── One instance per process, keyed by config ────────────────────────────────

_INSTANCES: dict[str, LayaTierClassifier] = {}
_INSTANCES_LOCK = threading.Lock()

# Keys that only steer how the router uses the tier, not the classifier itself.
_ROUTER_ONLY_KEYS = ("mode", "tiers", "min_confidence", "max_chars")


def get_classifier(cfg: dict) -> LayaTierClassifier:
    """Return the shared classifier for this config, building it once."""
    key_cfg = {k: v for k, v in merge_config(cfg).items() if k not in _ROUTER_ONLY_KEYS}
    key = json.dumps(key_cfg, sort_keys=True, default=str)
    with _INSTANCES_LOCK:
        inst = _INSTANCES.get(key)
        if inst is None:
            inst = LayaTierClassifier(cfg)
            _INSTANCES[key] = inst
        return inst
