import logging
import os
import re
from pathlib import Path

from optmod.routing import BaseRouter
from optmod.routing.context import RoutingContext
from optmod.schemas import RoutingDecision

_WEIGHTS_PATH = Path(__file__).parent / "trouter_weights.pt"
_DEFAULT_COST_WEIGHT = 0.3


def _normalize(s: str) -> str:
    """Lowercase and collapse all separators (/ : - _) to underscore."""
    return re.sub(r"[/_:\-]", "_", s.lower())


def _resolve_model(idx: int, model_ids: list[str], registry):
    """
    Map a TRouter model index to a ModelConfig by name matching.

    The checkpoint uses model_ids like 'arcee-ai_trinity-large-thinking_free'
    and 'deepseek-v4-flash'; the registry may use names like
    'arcee-ai/trinity-large-thinking:free' or 'deepseek/deepseek-v4-flash'.

    Matching strategy (first hit wins):
      1. Exact match on full normalised name.
      2. Exact match on just the local component after the last '/' in the
         registry name (handles 'deepseek/deepseek-v4-flash' → 'deepseek-v4-flash').
      3. Prefix match in either direction on the full name.
    """
    if idx >= len(model_ids):
        return registry.primary

    norm_wid = _normalize(model_ids[idx])
    all_models = registry.all()

    # Pass 1: exact full-name match
    for m in all_models:
        if _normalize(m.name) == norm_wid:
            return m

    # Pass 2: match against local component (after last '/')
    for m in all_models:
        local = _normalize(m.name.split("/")[-1])
        if local == norm_wid:
            return m

    # Pass 3: prefix match on full name in either direction
    for m in all_models:
        norm_name = _normalize(m.name)
        if norm_wid.startswith(norm_name) or norm_name.startswith(norm_wid):
            return m

    return registry.primary


class TRouterRouter(BaseRouter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._ready = False
        self._model_ids: list[str] = []

        alpha_cfg = config.get("trouter_cost_weight")
        alpha_env = os.environ.get("TROUTER_COST_WEIGHT")
        self._cost_weight = float(alpha_cfg or alpha_env or _DEFAULT_COST_WEIGHT)

        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from optmod.routing.trouter.train_trouter import build_model
            from optmod.routing.trouter.train_trouter import route as _trouter_route

            ckpt = torch.load(_WEIGHTS_PATH, weights_only=False, map_location="cpu")
            cfg = ckpt["config"]

            self._nn = build_model(
                embed_dim=cfg["embed_dim"],
                num_task_types=cfg["num_task_types"],
                num_models=cfg["num_models"],
                hidden_dim=cfg["hidden_dim"],
            )
            self._nn.load_state_dict(ckpt["model_state_dict"])
            self._nn.eval()

            self._prior = ckpt["prior"]
            self._model_ids = ckpt.get("model_ids", [])
            encoder_name = ckpt.get("encoder_name", "all-MiniLM-L6-v2")
            self._encoder = SentenceTransformer(encoder_name)
            self._route_fn = _trouter_route
            self._torch = torch
            self._ready = True
            logging.info(
                f"[optmod] TRouterRouter loaded (α={self._cost_weight}, "
                f"encoder={encoder_name}, model_ids={self._model_ids})"
            )
        except Exception as exc:
            logging.warning(f"[optmod] TRouterRouter failed to load: {exc}")

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._passthrough(ctx, "trouter: not initialised, using primary")
        try:
            return self._route(ctx)
        except Exception as exc:
            logging.warning(f"[optmod] TRouterRouter.route error: {exc}")
            return self._passthrough(ctx, f"trouter error: {exc}")

    def _route(self, ctx: RoutingContext) -> RoutingDecision:
        text = ctx.features.last_user_message or ""
        logging.info(f"[trouter] encoding text ({len(text)} chars): {text[:120]!r}")
        logging.info(f"[trouter] ctx.features fields: {vars(ctx.features)}")
        
        emb = self._encoder.encode(text, convert_to_tensor=True).cpu()  # [384]

        chosen_idx, scores_adj, _ = self._route_fn(
            self._nn, emb, self._prior, self._cost_weight
        )
        chosen_idx = int(chosen_idx)

        model = _resolve_model(chosen_idx, self._model_ids, ctx.registry)
        mutator = "thinking_mode" if model.thinking_mode else "noop"
        adj_score = float(scores_adj[chosen_idx].item())
        confidence = float(
            self._torch.softmax(scores_adj, dim=0)[chosen_idx].item()
        )
        weight_id = self._model_ids[chosen_idx] if chosen_idx < len(self._model_ids) else "?"

        return RoutingDecision(
            model=model,
            mutator=mutator,
            reason=(
                f"trouter: idx={chosen_idx} ({weight_id}) → {model.name} "
                f"score={adj_score:.3f} α={self._cost_weight}"
            ),
            confidence=confidence,
            router_name=self.name,
        )
