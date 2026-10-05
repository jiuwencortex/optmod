#!/usr/bin/env python3
"""
train_trouter.py
================
Trains TRouter on the tensor dataset produced by build_training_tensors.py.

Architecture
────────────
TRouter has three learned components:

  A) Task Classifier
     query_embedding [embed_dim]
     → Linear(embed_dim, 128) → ReLU → Linear(128, T) → Softmax
     → task_type_probs [T]

     Trained with cross-entropy against the LLM-assigned task_type_ids.
     Prior regularisation nudges uncertain predictions toward the observed
     task-type frequency distribution (the prior tensor in the dataset).

  B) Performance Heads  (one per model)
     [query_embedding | task_type_probs]  [embed_dim + T]
     → Linear(embed_dim+T, 64) → ReLU → Linear(64, 1) → Sigmoid
     → predicted_score  [0, 1]

     Trained with MSE against overall_score from WildClawBench.

  C) Cost Heads  (one per model)
     same input as B
     → Linear(embed_dim+T, 64) → ReLU → Linear(64, 1) → Softplus
     → predicted_cost  [≥ 0]

     Trained with MSE against cost_usd_computed.
     All-zero for local models — cost heads learn to output ~0 for those.

Routing decision at inference time
────────────────────────────────────
Given a new query embedding and a cost_weight α ∈ [0, 1]:

  score_i = predicted_score_i - α * normalised_cost_i
  route to model = argmax(score_i)

α = 0.0 → always pick highest quality  (ignore cost)
α = 1.0 → maximise quality-per-dollar  (strong cost pressure)

Usage
─────
  pip install torch
  python3 train_trouter.py --dataset training_dataset.pt
  python3 train_trouter.py --dataset training_dataset.pt \\
      --epochs 200 --lr 1e-3 --cost-weight 0.3 --out trouter_weights.pt
"""

import argparse
import sys
from pathlib import Path


# ── Model definition ──────────────────────────────────────────────────────────

def build_model(embed_dim: int, num_task_types: int, num_models: int,
                hidden_dim: int = 128):
    """
    Returns the TRouter nn.Module.
    Defined as a function (not a class at module level) so the file can be
    imported without requiring torch at import time.
    """
    import torch
    import torch.nn as nn

    class TRouter(nn.Module):
        def __init__(self):
            super().__init__()

            # A: Task classifier
            self.task_classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_task_types),
            )
            # Softmax applied in forward() so we can use CrossEntropyLoss
            # (which expects raw logits) during training and softmax at inference

            # B + C: Per-model performance and cost heads
            head_input_dim = embed_dim + num_task_types
            self.perf_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),       # output in [0, 1]
                )
                for _ in range(num_models)
            ])
            self.cost_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Softplus(),      # output ≥ 0
                )
                for _ in range(num_models)
            ])

        def forward(self, query_emb, prior=None, prior_weight=0.2):
            """
            Args:
                query_emb   : [batch, embed_dim]
                prior       : [num_task_types]  — dataset-level frequency prior
                prior_weight: float — how much to pull predictions toward prior
                              0.0 = pure classifier, 1.0 = always use prior

            Returns:
                task_logits : [batch, T]   — raw logits for cross-entropy loss
                task_probs  : [batch, T]   — softmax probabilities
                perf_preds  : [batch, M]   — predicted scores per model
                cost_preds  : [batch, M]   — predicted costs per model
            """
            import torch
            import torch.nn.functional as F

            task_logits = self.task_classifier(query_emb)   # [B, T]
            task_probs  = F.softmax(task_logits, dim=-1)    # [B, T]

            # Prior regularisation:
            # Blend the classifier's prediction toward the observed prior.
            # This is the "cold-start" protection from the TRouter paper —
            # when the classifier is uncertain, fall back to what's common.
            if prior is not None:
                prior_expanded = prior.unsqueeze(0).expand_as(task_probs)
                task_probs = (
                    (1.0 - prior_weight) * task_probs
                    + prior_weight       * prior_expanded
                )

            combined = torch.cat([query_emb, task_probs], dim=-1)  # [B, embed+T]

            perf_preds = torch.cat(
                [head(combined) for head in self.perf_heads], dim=-1
            )  # [B, M]
            cost_preds = torch.cat(
                [head(combined) for head in self.cost_heads], dim=-1
            )  # [B, M]

            return task_logits, task_probs, perf_preds, cost_preds

    return TRouter()


# ── Routing decision (used at inference and in simulation) ────────────────────

def route(model, query_emb, prior, cost_weight: float = 0.3):
    """
    Given a query embedding, return the index of the model to route to.

    cost_weight α:
      0.0 → always pick best quality (never mind cost)
      0.3 → default — gentle cost pressure
      1.0 → maximise quality-per-dollar aggressively

    Returns:
        chosen_model_idx : int
        scores_adj       : Tensor[M] — adjusted scores used for decision
        task_probs       : Tensor[T] — task type distribution
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        _, task_probs, perf_preds, cost_preds = model(
            query_emb, prior=prior, prior_weight=0.2
        )
        perf = perf_preds.squeeze(0)   # [M]
        cost = cost_preds.squeeze(0)   # [M]

        # Normalise cost to [0, 1] range for stable blending
        cost_max = cost.max()
        cost_norm = cost / (cost_max + 1e-8)

        scores_adj = perf - cost_weight * cost_norm
        chosen = scores_adj.argmax().item()

    return chosen, scores_adj, task_probs.squeeze(0)


# ── Training ──────────────────────────────────────────────────────────────────

def train(dataset_path: Path, out_path: Path, epochs: int, lr: float,
          cost_weight: float, prior_weight: float, lambda_cost: float,
          hidden_dim: int, device_str: str, seed: int):

    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(seed)
    device = torch.device(device_str)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {dataset_path}")
    ds = torch.load(dataset_path, weights_only=False)

    query_emb    = ds["query_embeddings"].to(device)   # [N, embed_dim]
    task_type_ids = ds["task_type_ids"].to(device)      # [N]
    scores       = ds["scores"].to(device)              # [N, M]
    costs        = ds["costs"].to(device)               # [N, M]
    prior        = ds["prior"].to(device)               # [T]

    model_ids        = ds["model_ids"]
    task_type_labels = ds["task_type_labels"]
    task_names       = ds["task_names"]

    N, embed_dim  = query_emb.shape
    M             = scores.shape[1]
    T             = prior.shape[0]

    print(f"  Tasks={N}, Models={M}, TaskTypes={T}, EmbedDim={embed_dim}")
    print(f"  Models: {model_ids}")
    print(f"  Task types: {task_type_labels}")

    # ── Build model ───────────────────────────────────────────────────────────
    trouter = build_model(embed_dim, T, M, hidden_dim).to(device)
    total_params = sum(p.numel() for p in trouter.parameters())
    print(f"\nTRouter parameters: {total_params:,}")

    optimizer = optim.Adam(trouter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    # Dataset is small (60 tasks) — train on full batch each epoch
    print(f"\nTraining for {epochs} epochs  (lr={lr}, cost_weight={cost_weight}, "
          f"prior_weight={prior_weight})\n")
    print(f"{'epoch':>6}  {'loss_total':>11}  {'loss_task':>10}  "
          f"{'loss_perf':>10}  {'loss_cost':>10}  {'lr':>8}")
    print("─" * 65)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        trouter.train()
        optimizer.zero_grad()

        task_logits, task_probs, perf_preds, cost_preds = trouter(
            query_emb, prior=prior, prior_weight=prior_weight
        )

        # Loss A: task type classification (cross-entropy on logits)
        loss_task = ce_loss(task_logits, task_type_ids)

        # Loss B: performance prediction (MSE per model, averaged)
        loss_perf = mse_loss(perf_preds, scores)

        # Loss C: cost prediction (MSE per model, averaged)
        # Scale costs to [0,1] so MSE is comparable to performance MSE
        cost_max  = costs.max().clamp(min=1e-8)
        loss_cost = mse_loss(cost_preds, costs / cost_max)

        # Combined loss — lambda_cost controls how much cost accuracy matters
        loss = loss_task + loss_perf + lambda_cost * loss_cost
        loss.backward()

        # Gradient clipping — prevents exploding gradients on small datasets
        nn.utils.clip_grad_norm_(trouter.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>11.6f}  {loss_task.item():>10.6f}  "
                  f"{loss_perf.item():>10.6f}  {loss_cost.item():>10.6f}  "
                  f"{current_lr:>8.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() if hasattr(v, 'clone') else v
                          for k, v in trouter.state_dict().items()}

    print(f"\nBest loss: {best_loss:.6f}")

    # ── Post-training: routing simulation on training data ────────────────────
    trouter.load_state_dict(best_state)
    trouter.eval()

    print("\n── Routing simulation on training data ───────────────────────────────")
    print("(This shows what TRouter would have decided on each training task)")
    print(f"\n  {'task_name':<42} {'routed_to':<20} {'routed_score':>12} "
          f"{'best_score':>10} {'gap':>6} {'cost_saved':>10}")
    print("  " + "─" * 104)

    total_routed_score  = 0.0
    total_best_score    = 0.0
    total_cost_routed   = 0.0
    total_cost_best     = 0.0
    correct_routes      = 0

    import torch.nn.functional as F

    with torch.no_grad():
        for i in range(N):
            emb = query_emb[i]
            chosen_idx, _, _ = route(trouter, emb, prior, cost_weight)

            routed_score = scores[i, chosen_idx].item()
            routed_cost  = costs[i, chosen_idx].item()
            best_idx     = scores[i].argmax().item()
            best_score   = scores[i, best_idx].item()
            best_cost    = costs[i, best_idx].item()
            gap          = best_score - routed_score

            total_routed_score += routed_score
            total_best_score   += best_score
            total_cost_routed  += routed_cost
            total_cost_best    += best_cost
            if chosen_idx == best_idx or gap <= 0.10:
                correct_routes += 1

            routed_model_short = model_ids[chosen_idx].split("_")[0]
            print(f"  {task_names[i]:<42} {routed_model_short:<20} "
                  f"{routed_score:>12.3f} {best_score:>10.3f} "
                  f"{gap:>+6.3f} ${routed_cost:>9.6f}")

    avg_routed = total_routed_score / N
    avg_best   = total_best_score / N
    pct_correct = correct_routes / N * 100
    cost_saved  = total_cost_best - total_cost_routed

    print(f"\n  Avg routed score      : {avg_routed:.3f}")
    print(f"  Avg best-model score  : {avg_best:.3f}")
    print(f"  Quality retention     : {avg_routed/avg_best*100:.1f}%")
    print(f"  Acceptable routes     : {correct_routes}/{N} ({pct_correct:.0f}%)")
    print(f"  Cost saved vs always-best : ${cost_saved:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict = {
        # Model weights
        "model_state_dict": best_state,

        # Architecture config — needed to reconstruct at inference time
        "config": {
            "embed_dim":       embed_dim,
            "num_task_types":  T,
            "num_models":      M,
            "hidden_dim":      hidden_dim,
        },

        # Metadata
        "model_ids":          model_ids,
        "task_type_labels":   task_type_labels,
        "prior":              prior.cpu(),
        "encoder_name":       ds.get("encoder_name", "all-MiniLM-L6-v2"),

        # Training config — for reproducibility
        "training": {
            "epochs":       epochs,
            "lr":           lr,
            "cost_weight":  cost_weight,
            "prior_weight": prior_weight,
            "lambda_cost":  lambda_cost,
            "best_loss":    best_loss,
            "seed":         seed,
        },
    }

    torch.save(save_dict, out_path)
    print(f"\nSaved TRouter weights to: {out_path}")
    print(f"Done. Next step: python3 simulate_routing.py "
          f"--weights {out_path.name} --dataset training_dataset.pt")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train TRouter on WildClawBench tensor dataset"
    )
    parser.add_argument("--dataset",      default="training_dataset.pt")
    parser.add_argument("--out",          default="trouter_weights.pt")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--cost-weight",  type=float, default=0.3,
                        help="α: 0=quality-only, 1=cost-aggressive (default 0.3)")
    parser.add_argument("--prior-weight", type=float, default=0.2,
                        help="How much to blend classifier toward prior (default 0.2)")
    parser.add_argument("--lambda-cost",  type=float, default=0.1,
                        help="Weight of cost prediction loss (default 0.1)")
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. Run: pip install torch", file=sys.stderr)
        sys.exit(1)

    train(
        dataset_path  = Path(args.dataset).expanduser(),
        out_path      = Path(args.out).expanduser(),
        epochs        = args.epochs,
        lr            = args.lr,
        cost_weight   = args.cost_weight,
        prior_weight  = args.prior_weight,
        lambda_cost   = args.lambda_cost,
        hidden_dim    = args.hidden_dim,
        device_str    = args.device,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()