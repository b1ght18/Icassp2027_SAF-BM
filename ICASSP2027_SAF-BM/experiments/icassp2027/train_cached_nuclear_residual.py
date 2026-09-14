from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .boundary_residual import FixedRankBoundaryResidual
    from .train_cached_rank_candidates import class_weights
    from .train_dcase_stage import macro_accuracy, set_seed
except ImportError:
    from boundary_residual import FixedRankBoundaryResidual
    from train_cached_rank_candidates import class_weights
    from train_dcase_stage import macro_accuracy, set_seed


def weighted_ce(
    features: torch.Tensor,
    labels: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    delta_weight: torch.Tensor,
    delta_bias: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(features, source_weight + delta_weight, source_bias + delta_bias)
    return F.cross_entropy(logits, labels, weight=weights)


def singular_value_threshold(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    shrunk = (singular - threshold).clamp_min(0.0)
    return (left * shrunk.unsqueeze(0)) @ right


def exact_rank(matrix: torch.Tensor, relative_tolerance: float = 1e-6) -> int:
    singular = torch.linalg.svdvals(matrix)
    if float(singular.max()) == 0.0:
        return 0
    return int((singular > singular.max() * relative_tolerance).sum().item())


def validation_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    delta_weight: torch.Tensor,
    delta_bias: torch.Tensor,
) -> tuple[float, float]:
    with torch.inference_mode():
        logits = F.linear(features, source_weight + delta_weight, source_bias + delta_bias)
        loss = float(F.cross_entropy(logits, labels))
        accuracy = macro_accuracy(labels.tolist(), logits.argmax(1).tolist())
    return loss, accuracy


def to_fixed_state(
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    delta_weight: torch.Tensor,
    delta_bias: torch.Tensor,
    alpha: float,
) -> tuple[int, dict[str, torch.Tensor]]:
    left, singular, right = torch.linalg.svd(delta_weight, full_matrices=False)
    keep = singular > singular.max().clamp_min(1e-20) * 1e-6
    rank = int(keep.sum().item())
    if rank == 0:
        # The fixed residual representation requires rank >= 1; a zero update is
        # represented by zero B and does not change the reported numerical rank.
        rank = 1
        model = FixedRankBoundaryResidual(source_weight, source_bias, rank, alpha)
        with torch.no_grad():
            model.B.zero_()
            model.delta_bias.copy_(delta_bias)
        return 1, copy.deepcopy(model.state_dict())
    scaling = alpha / rank
    root = torch.sqrt(singular[keep] / scaling)
    model = FixedRankBoundaryResidual(source_weight, source_bias, rank, alpha)
    with torch.no_grad():
        model.B.copy_(left[:, keep] * root.unsqueeze(0))
        model.A.copy_(root.unsqueeze(1) * right[keep])
        model.delta_bias.copy_(delta_bias)
    if not torch.allclose(model.delta_weight(), delta_weight, atol=2e-5, rtol=2e-5):
        raise RuntimeError("SVD compression changed the learned boundary residual")
    return rank, copy.deepcopy(model.state_dict())


def optimize_lambda(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    weights: torch.Tensor,
    nuclear_lambda: float,
    iterations: int,
    patience: int,
    initial_lipschitz: float,
) -> dict[str, object]:
    delta_weight = torch.zeros_like(source_weight)
    delta_bias = torch.zeros_like(source_bias)
    lipschitz = float(initial_lipschitz)
    best: dict[str, object] | None = None
    stale = 0
    for iteration in range(1, iterations + 1):
        current_weight = delta_weight.detach().requires_grad_(True)
        current_bias = delta_bias.detach().requires_grad_(True)
        smooth = weighted_ce(
            fit_features,
            fit_labels,
            source_weight,
            source_bias,
            current_weight,
            current_bias,
            weights,
        )
        gradient_weight, gradient_bias = torch.autograd.grad(
            smooth, (current_weight, current_bias)
        )
        accepted = False
        for _ in range(30):
            candidate_weight = singular_value_threshold(
                current_weight - gradient_weight / lipschitz,
                nuclear_lambda / lipschitz,
            )
            candidate_bias = current_bias - gradient_bias / lipschitz
            with torch.inference_mode():
                candidate_smooth = weighted_ce(
                    fit_features,
                    fit_labels,
                    source_weight,
                    source_bias,
                    candidate_weight,
                    candidate_bias,
                    weights,
                )
                difference_weight = candidate_weight - current_weight
                difference_bias = candidate_bias - current_bias
                quadratic = (
                    smooth.detach()
                    + (gradient_weight * difference_weight).sum()
                    + (gradient_bias * difference_bias).sum()
                    + 0.5
                    * lipschitz
                    * (difference_weight.square().sum() + difference_bias.square().sum())
                )
            if float(candidate_smooth) <= float(quadratic) + 1e-7:
                accepted = True
                break
            lipschitz *= 2.0
        if not accepted:
            raise RuntimeError("Proximal-gradient backtracking failed")
        delta_weight = candidate_weight.detach()
        delta_bias = candidate_bias.detach()
        lipschitz = max(initial_lipschitz, lipschitz * 0.9)
        validation_loss, validation_accuracy = validation_metrics(
            validation_features,
            validation_labels,
            source_weight,
            source_bias,
            delta_weight,
            delta_bias,
        )
        candidate = (validation_accuracy, -validation_loss, -iteration)
        if best is None or candidate > best["selection_key"]:
            numerical_rank = exact_rank(delta_weight)
            best = {
                "selection_key": candidate,
                "delta_weight": delta_weight.clone(),
                "delta_bias": delta_bias.clone(),
                "validation_loss": validation_loss,
                "validation_macro_accuracy": validation_accuracy,
                "best_iteration": iteration,
                "training_loss": float(candidate_smooth),
                "nuclear_norm": float(torch.linalg.svdvals(delta_weight).sum()),
                "rank": max(1, numerical_rank),
                "numerical_rank": numerical_rank,
                "lipschitz": lipschitz,
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convex nuclear-norm boundary migration on frozen cached features"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--nuclear-lambdas", default="0,0.0001,0.0003,0.001,0.003,0.01,0.03"
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--initial-lipschitz", type=float, default=1.0)
    parser.add_argument("--selection-tolerance", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1193)
    args = parser.parse_args()
    set_seed(args.seed)
    data = np.load(args.cache, allow_pickle=False)
    fit_features = torch.from_numpy(data["fit_features"].astype(np.float32))
    fit_labels = torch.from_numpy(data["fit_labels"].astype(np.int64))
    validation_features = torch.from_numpy(data["validation_features"].astype(np.float32))
    validation_labels = torch.from_numpy(data["validation_labels"].astype(np.int64))
    source_weight = torch.from_numpy(data["source_weight"].astype(np.float32))
    source_bias = torch.from_numpy(data["source_bias"].astype(np.float32))
    weights = class_weights(data["fit_labels"])

    candidates = []
    for nuclear_lambda in [float(value) for value in args.nuclear_lambdas.split(",")]:
        result = optimize_lambda(
            fit_features,
            fit_labels,
            validation_features,
            validation_labels,
            source_weight,
            source_bias,
            weights,
            nuclear_lambda,
            args.iterations,
            args.patience,
            args.initial_lipschitz,
        )
        result["nuclear_lambda"] = nuclear_lambda
        candidates.append(result)
        printable = {key: value for key, value in result.items() if key not in {
            "selection_key", "delta_weight", "delta_bias"
        }}
        print(json.dumps(printable, sort_keys=True), flush=True)

    by_rank: dict[int, dict[str, object]] = {}
    for candidate in candidates:
        rank = int(candidate["rank"])
        key = (
            float(candidate["validation_macro_accuracy"]),
            -float(candidate["validation_loss"]),
        )
        if rank not in by_rank or key > by_rank[rank]["selection_key"]:
            by_rank[rank] = {**candidate, "selection_key": key}
    best_accuracy = max(float(value["validation_macro_accuracy"]) for value in by_rank.values())
    eligible = [
        value
        for value in by_rank.values()
        if float(value["validation_macro_accuracy"])
        >= best_accuracy - args.selection_tolerance
    ]
    selected = min(
        eligible,
        key=lambda value: (
            int(value["rank"]),
            -float(value["validation_macro_accuracy"]),
        ),
    )
    states: dict[int, dict[str, torch.Tensor]] = {}
    rows = []
    for rank, value in sorted(by_rank.items()):
        encoded_rank, state = to_fixed_state(
            source_weight,
            source_bias,
            value["delta_weight"],
            value["delta_bias"],
            args.alpha,
        )
        if encoded_rank != rank:
            raise RuntimeError("Rank changed during fixed-factor encoding")
        states[rank] = state
        rows.append(
            {
                key: value[key]
                for key in (
                    "rank",
                    "numerical_rank",
                    "validation_macro_accuracy",
                    "validation_loss",
                    "best_iteration",
                    "training_loss",
                    "nuclear_norm",
                    "nuclear_lambda",
                    "lipschitz",
                )
            }
        )
    selected_rank = int(selected["rank"])
    payload = {
        "format": "icassp2027-nuclear-boundary-rank-v1",
        "selected_rank": selected_rank,
        "selected_state_dict": states[selected_rank],
        "all_states": states,
        "rows": rows,
        "selection_tolerance": args.selection_tolerance,
        "nuclear_lambdas": [float(value) for value in args.nuclear_lambdas.split(",")],
        "iterations": args.iterations,
        "patience": args.patience,
        "initial_lipschitz": args.initial_lipschitz,
        "alpha": args.alpha,
        "loss": "inverse-frequency-class-balanced-cross-entropy",
        "task": int(data["task"]),
        "stage": int(data["stage"]),
        "seed": args.seed,
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    frame = pd.DataFrame(rows)
    frame["selected"] = frame["rank"] == selected_rank
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary, index=False)
    print(
        json.dumps(
            {
                "best_validation": best_accuracy,
                "selected_rank": selected_rank,
                "selected_lambda": selected["nuclear_lambda"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
