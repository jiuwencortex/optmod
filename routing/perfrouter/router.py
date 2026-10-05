import inspect
import logging
import os
import re
import time
import json as _json
from pathlib import Path

from optmod.routing import BaseRouter
from optmod.routing.context import RoutingContext
from optmod.schemas import RoutingDecision
from optmod.routing.perfrouter.laya_tier import (
    DEFAULTS as _LAYA_DEFAULTS, TierResult, get_classifier, merge_config as _merge_laya_cfg,
    target as _laya_target,
)

_DIR = Path(__file__).parent
_DEFAULT_COST_WEIGHT = 0.3
_DEFAULT_BASELINE    = "deepseek/deepseek-v4-pro"
_LAYA_MODES          = ("off", "shadow", "active")


def resolve_tier_params(
    result: TierResult | None, status: str, mode: str, laya_cfg: dict,
    static_delta: float, static_cap: float | None,
) -> tuple[float, float | None, bool]:
    """
    Pick (delta, cost_cap, applied) for one request.

    Tier values apply only in active mode, with an `ok` answer at or above
    min_confidence, and a configured entry for that tier. Otherwise the static
    values come back with applied=False. cost_cap None means no cap.
    """
    if mode != "active" or status != "ok" or result is None:
        return static_delta, static_cap, False
    min_conf = float(laya_cfg.get("min_confidence", _LAYA_DEFAULTS["min_confidence"]))
    if result.confidence < min_conf:
        return static_delta, static_cap, False
    tier_cfg = (laya_cfg.get("tiers") or {}).get(result.tier)
    if not isinstance(tier_cfg, dict):
        return static_delta, static_cap, False
    delta = float(tier_cfg.get("delta", static_delta))
    cap_raw = tier_cfg.get("cost_cap", static_cap)
    cap = float(cap_raw) if cap_raw is not None else None
    return delta, cap, True


def _normalize(s: str) -> str:
    """Lowercase and collapse all separators (/ : - _) to underscore."""
    return re.sub(r"[/_:\-]", "_", s.lower())


def _resolve_model(idx: int, model_ids: list[str], registry):
    """
    Map a model index to a ModelConfig by name matching.

    Copied verbatim from trouter/router.py — same matching strategy,
    same fallback to registry.primary.
    """
    if idx >= len(model_ids):
        return registry.primary

    norm_wid  = _normalize(model_ids[idx])
    all_models = registry.all()

    for m in all_models:
        if _normalize(m.name) == norm_wid:
            return m

    for m in all_models:
        local = _normalize(m.name.split("/")[-1])
        if local == norm_wid:
            return m

    for m in all_models:
        norm_name = _normalize(m.name)
        if norm_wid.startswith(norm_name) or norm_name.startswith(norm_wid):
            return m

    # Pass 4: match local component of trained name against registry full name.
    # Handles trained="deepseek/deepseek-v4-flash" → local="deepseek-v4-flash"
    # → registry entry "deepseek-v4-flash" (direct API, no prefix).
    if "/" in model_ids[idx]:
        local_wid = _normalize(model_ids[idx].split("/")[-1])
        for m in all_models:
            if _normalize(m.name) == local_wid:
                return m

    return registry.primary


def _extract_content(raw: str) -> str:
    """
    Extract actual user text from optmod's JSON-wrapped session message format.

    Messages arrive as:
      '{"source": "sess", ..., "content": "hi", "type": "user input"}'

    Falls back to raw string if not JSON or no content field.
    """
    if raw.strip().startswith("{"):
        try:
            return _json.loads(raw).get("content", raw)
        except (_json.JSONDecodeError, AttributeError):
            pass
    return raw


def _resolve_data_paths(perf_cfg: dict) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None]:
    """
    Resolve (router, taxonomy, registry, features, models_yaml) from config.

    Returns None for each path if not found in the configured location.
    Falls back to local _DIR copies with a deprecation warning when canonical
    paths exist but some files are missing.
    """
    data_dir_str = perf_cfg.get("perfrouter_data_dir")
    if not data_dir_str:
        return None, None, None, None, None

    data_dir = Path(data_dir_str).resolve()
    taxonomy  = data_dir / "task_taxonomy.json"
    registry  = data_dir / "model_registry.json"
    features  = data_dir / "model_features.csv"

    # models.yaml for runtime pricing
    models_yaml_str = perf_cfg.get("perfrouter_models_yaml")
    models_yaml = Path(models_yaml_str).resolve() if models_yaml_str else data_dir.parent / "models.yaml"

    # Checkpoint: auto → prefer data_dir/../models/, fall back to data_dir/
    ckpt_cfg = perf_cfg.get("checkpoint", "auto")
    if ckpt_cfg == "auto":
        candidate_models = data_dir.parent / "models" / "perf_router.pkl"
        candidate_data   = data_dir / "perf_router.pkl"
        if candidate_models.exists():
            router_path = candidate_models
        elif candidate_data.exists():
            router_path = candidate_data
        else:
            router_path = None
    elif ckpt_cfg:
        router_path = Path(ckpt_cfg).resolve()
        if not router_path.exists():
            router_path = None
    else:
        router_path = None

    if router_path and router_path.exists() and taxonomy.exists() and registry.exists() and features.exists():
        return router_path, taxonomy, registry, features, models_yaml

    found = [p for p in (taxonomy, registry, features) if p.exists()]
    if found:
        logging.warning(
            "[optmod] PerfRouter: not all canonical data files present in %s "
            "(found %d/3). Falling back to local copies in %s.",
            data_dir, len(found), _DIR,
        )
    return None, None, None, None, None


class PerfRouterRouter(BaseRouter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._ready = False

        perf_cfg = config.get("perf_router", {})

        # ── Policy parameters (nested config with flat-key fallback) ──────────
        alpha_cfg = perf_cfg.get("cost_weight") or config.get("perf_router_cost_weight")
        if config.get("perf_router_cost_weight") and not perf_cfg.get("cost_weight"):
            logging.warning("[optmod] Deprecated flat key perf_router_cost_weight — use perf_router.cost_weight")
        alpha_env = os.environ.get("PERF_ROUTER_COST_WEIGHT")
        self._cost_weight = float(alpha_cfg or alpha_env or _DEFAULT_COST_WEIGHT)

        baseline_cfg = perf_cfg.get("baseline") or config.get("perf_router_baseline")
        baseline_env = os.environ.get("PERF_ROUTER_BASELINE")
        self._baseline = baseline_cfg or baseline_env or _DEFAULT_BASELINE

        threshold_cfg = (perf_cfg.get("degradation_threshold")
                         if "degradation_threshold" in perf_cfg
                         else config.get("perf_router_degradation_threshold"))
        threshold_env = os.environ.get("PERF_ROUTER_DEGRADATION_THRESHOLD")
        self._degradation_threshold = float(
            threshold_cfg if threshold_cfg is not None else (threshold_env or 0.0)
        )

        min_sim_cfg = (perf_cfg.get("min_similarity")
                       if "min_similarity" in perf_cfg
                       else config.get("perf_router_min_similarity"))
        min_sim_env = os.environ.get("PERF_ROUTER_MIN_SIMILARITY")
        self._min_similarity = float(
            min_sim_cfg if min_sim_cfg is not None else (min_sim_env or 0.20)
        )

        pin_cfg = config.get("session_pin", {})
        bonus_cfg = (pin_cfg.get("soft_bonus_weight")
                     or config.get("session_pin_soft_bonus_weight"))
        bonus_env = os.environ.get("SESSION_PIN_SOFT_BONUS_WEIGHT")
        self._soft_bonus_weight = float(
            bonus_cfg if bonus_cfg is not None else (bonus_env or 0.5)
        )

        # cost_cap_multiplier: YAML null → Python None → pass as float("inf") to inference
        cap_raw = perf_cfg.get("cost_cap_multiplier", 2.0)
        self._cost_cap_multiplier = cap_raw  # None means cap disabled

        # ── Laya difficulty tier (optional) ───────────────────────────────────
        laya_raw = perf_cfg.get("laya") or {}
        self._laya_cfg: dict = _merge_laya_cfg(laya_raw)
        self._laya_mode = str(self._laya_cfg.get("mode") or "off")
        if self._laya_mode not in _LAYA_MODES:
            logging.warning("[optmod] Unknown perf_router.laya.mode %r — using off", self._laya_mode)
            self._laya_mode = "off"
        for tier_name, tier_cfg in (self._laya_cfg.get("tiers") or {}).items():
            if isinstance(tier_cfg, dict) and float(tier_cfg.get("delta", 1.0)) <= 0.0:
                logging.warning(
                    "[optmod] perf_router.laya.tiers.%s.delta <= 0 switches route() to "
                    "argmax(utility) mode, not 'best quality'. Use a small positive δ.",
                    tier_name,
                )
        self._laya = None
        if self._laya_mode != "off":
            self._ensure_laya()

        # ── Data path resolution ──────────────────────────────────────────────
        router_path, taxonomy_path, registry_path, features_path, models_yaml_path = \
            _resolve_data_paths(perf_cfg)

        # Fall back to local copies (backward compat for existing deployments)
        if router_path is None:
            router_path    = _DIR / "perf_router.pkl"
            taxonomy_path  = _DIR / "task_taxonomy.json"
            registry_path  = _DIR / "model_registry.json"
            features_path  = _DIR / "model_features.csv"
            models_yaml_path = _DIR / "models.yaml"

        try:
            # Prefer perfrouter package if installed; fall back to local copy
            try:
                from perfrouter.inference.perf_router_inference import PerfRouterInference
            except ImportError:
                from optmod.routing.perfrouter.inference import PerfRouterInference

            # Pass models_yaml_path only if the class supports it
            init_kwargs: dict = {
                "router_path":              router_path,
                "taxonomy_path":            taxonomy_path,
                "registry_path":            registry_path,
                "features_path":            features_path,
                "cost_weight":              self._cost_weight,
                "baseline_model":           self._baseline,
                "min_similarity_threshold": self._min_similarity,
            }
            if "models_yaml_path" in inspect.signature(PerfRouterInference.__init__).parameters:
                init_kwargs["models_yaml_path"] = models_yaml_path

            self._perf_router = PerfRouterInference(**init_kwargs)
            self._ready = True
            cap_str = str(self._cost_cap_multiplier) if self._cost_cap_multiplier is not None else "none"
            logging.info(
                "[optmod] PerfRouterRouter loaded "
                "(α=%s, baseline=%s, degradation=%s, min_sim=%s, cost_cap=%s)",
                self._cost_weight, self._baseline,
                self._degradation_threshold, self._min_similarity, cap_str,
            )
        except Exception as exc:
            logging.warning("[optmod] PerfRouterRouter failed to load: %s", exc)

    # ── Laya control ──────────────────────────────────────────────────────────

    def _ensure_laya(self) -> None:
        if self._laya is not None:
            return
        try:
            self._laya = get_classifier(self._laya_cfg)
        except Exception as exc:
            logging.warning("[optmod] Laya tier classifier failed to build: %s", exc)
            self._laya = None

    def set_laya_mode(self, mode: str) -> None:
        if mode not in _LAYA_MODES:
            raise ValueError(f"unknown laya mode: {mode}")
        self._laya_mode = mode
        if mode != "off":
            self._ensure_laya()

    def laya_status(self) -> dict:
        out = {
            "mode":       self._laya_mode,
            "backend":    self._laya_cfg.get("backend"),
            "target":     _laya_target(self._laya_cfg),
            "breaker":    "not_loaded",
        }
        if self._laya is not None:
            out.update(self._laya.status())
        return out

    def _find_inference_model_id(self, registry_name: str) -> str | None:
        """Inverse of _resolve_model: registry model name → inference _model_ids entry."""
        if not self._ready:
            return None
        norm_target = _normalize(registry_name)
        norm_local  = _normalize(registry_name.split("/")[-1])
        for mid in self._perf_router._model_ids:
            if _normalize(mid) == norm_target:
                return mid
        for mid in self._perf_router._model_ids:
            if _normalize(mid.split("/")[-1]) == norm_local:
                return mid
        return None

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._passthrough(ctx, "perf_router: not initialised, using primary")
        try:
            return self._route(ctx)
        except Exception as exc:
            logging.warning("[optmod] PerfRouterRouter.route error: %s", exc)
            return self._passthrough(ctx, f"perf_router error: {exc}")

    def _route(self, ctx: RoutingContext) -> RoutingDecision:
        token_count = ctx.features.token_count or None

        # ── Detect image attachments ──────────────────────────────────────────
        has_images = False
        for msg in ctx.request.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_images = True
                        break
            if has_images:
                break

        # ── Build routing text from last 3 user messages ──────────────────────
        # Always use the last 3 user messages rather than just the last message.
        # This correctly handles follow-ups like "yes", "fix it", "do that"
        # which are meaningless without context from prior turns.
        #
        # Only user messages are included — system prompts describe model
        # behaviour (not the task), assistant messages are prior responses,
        # and tool messages are structured JSON blobs. All three add noise
        # to the sentence-BERT classification.
        #
        # The last message is repeated at the end to bias the embedding
        # toward the current intent without losing surrounding context.
        user_messages = []
        for msg in ctx.request.messages:
            if msg.role != "user":
                continue
            if isinstance(msg.content, str) and msg.content.strip():
                content = _extract_content(msg.content.strip())
            elif isinstance(msg.content, list):
                content = " ".join(
                    _extract_content(p.get("text", ""))
                    for p in msg.content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            else:
                content = ""
            if content:
                user_messages.append(content)

        last_3 = user_messages[-3:]

        # Laya sees the same window without the repeated last message (the
        # repeat is a MiniLM weighting trick). Keep the tail: current turn is last.
        laya_text = "\n".join(last_3)
        max_chars = int(self._laya_cfg.get("max_chars") or 0)
        if max_chars > 0 and len(laya_text) > max_chars:
            laya_text = laya_text[-max_chars:]

        # Repeat last message to weight current intent in the embedding
        if last_3:
            last_3 = last_3 + [last_3[-1]]

        routing_text = "\n".join(last_3) if last_3 else ""

        # ── Resolve session pin to inference model_id ─────────────────────────
        pin_info = None
        if ctx.session_pin is not None:
            pin_mid = self._find_inference_model_id(ctx.session_pin.model_name)
            if pin_mid is not None:
                pin_info = {
                    "model_id":     pin_mid,
                    "cache_rate":   ctx.session_pin.last_cache_rate,
                    "bonus_weight": self._soft_bonus_weight,
                }

        # ── Laya difficulty tier ──────────────────────────────────────────────
        # Below min_similarity, route() takes fallback_ambiguous (cheapest
        # eligible model) and ignores δ, so the tier cannot act there yet. We
        # still classify and log it (observe-only) to measure how many hard
        # queries land in that fallback; pr_routing_mode marks those rows.
        mode        = self._laya_mode
        tier_result = None
        laya_status = "off"
        laya_ms     = 0.0
        low_sim     = False
        if mode != "off":
            if not routing_text:
                laya_status = "skipped_empty"
            else:
                top = self._perf_router.classify_task(routing_text)
                low_sim = not top or top[0][1] < self._min_similarity
                if self._laya is None:
                    laya_status = "unavailable"
                else:
                    t0 = time.perf_counter()
                    tier_result, laya_status = self._laya.classify(laya_text)
                    laya_ms = (time.perf_counter() - t0) * 1000

        static_delta = self._degradation_threshold
        static_cap   = (float(self._cost_cap_multiplier)
                        if self._cost_cap_multiplier is not None else None)
        delta, cap_val, applied = resolve_tier_params(
            tier_result, laya_status, "shadow" if low_sim else mode,
            self._laya_cfg, static_delta, static_cap,
        )
        if (laya_status == "ok" and tier_result is not None
                and tier_result.confidence < float(self._laya_cfg.get("min_confidence", 0.0))):
            laya_status = "low_confidence"

        # ── Cost cap: None → float("inf") (disabled) ─────────────────────────
        cap = cap_val if cap_val is not None else float("inf")

        # ── Route ─────────────────────────────────────────────────────────────
        decision = self._perf_router.route(
            routing_text,
            token_count           = token_count,
            has_images            = has_images,
            degradation_threshold = delta,
            pin_info              = pin_info,
            cost_cap_multiplier   = cap,
        )

        # ── Shadow: what would the tier values have picked? ───────────────────
        shadow_model = ""
        if mode == "shadow" and laya_status == "ok" and not low_sim:
            s_delta, s_cap, s_applied = resolve_tier_params(
                tier_result, laya_status, "active", self._laya_cfg, static_delta, static_cap,
            )
            if s_applied and (s_delta, s_cap) != (static_delta, static_cap):
                shadow = self._perf_router.route(
                    routing_text,
                    token_count           = token_count,
                    has_images            = has_images,
                    degradation_threshold = s_delta,
                    pin_info              = pin_info,
                    cost_cap_multiplier   = s_cap if s_cap is not None else float("inf"),
                )
                shadow_model = shadow.get("decision_model", "")

        chosen_id    = decision["decision_model"]
        model        = _resolve_model(0, [chosen_id], ctx.registry)
        mutator      = "thinking_mode" if model.thinking_mode else "noop"

        task_type      = decision.get("task_type", "unknown")
        cost_saved_pct = decision.get("cost_saved_pct", 0.0)
        alpha          = decision.get("alpha", self._cost_weight)
        quality        = decision.get("predicted_quality", 0.5)
        routing_mode   = decision.get("routing_mode", "normal")
        pin_bonus      = decision.get("pin_soft_bonus", 0.0)
        cap_str        = str(cap_val) if cap_val is not None else "none"
        tier_name      = tier_result.tier if tier_result is not None else ""

        return RoutingDecision(
            model       = model,
            mutator     = mutator,
            reason      = (
                f"perf_router: {task_type} → {chosen_id} "
                f"quality={quality:.3f} "
                f"cost_saved={cost_saved_pct:+.1f}% "
                f"α={alpha:.2f} "
                f"degradation={delta:.2f} "
                f"mode={routing_mode} "
                f"pin_soft_bonus={pin_bonus:.3f} "
                f"cost_cap={cap_str} "
                f"tier={tier_name or '-'} "
                f"laya={laya_status}"
            ),
            confidence  = float(quality),
            router_name = self.name,
            meta        = {
                "pr_task_type":       task_type,
                "pr_routing_mode":    routing_mode,
                "pr_top_similarity":  float(decision.get("top_similarity", 0.0)),
                "laya_status":        laya_status,
                "laya_tier":          tier_name,
                "laya_confidence":    float(tier_result.confidence) if tier_result is not None else 0.0,
                "laya_ms":            round(laya_ms, 2),
                "laya_cached":        bool(tier_result.cached) if tier_result is not None else False,
                "laya_applied":       applied,
                "laya_shadow_model":  shadow_model,
                "effective_delta":    float(delta),
                "effective_cost_cap": cap_val,
            },
        )
