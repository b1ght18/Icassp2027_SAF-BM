"""Harm-constrained rank-growing decision-boundary migration on cached features."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .boundary_residual import fix_softmax_gauge
    from .train_cached_crossing_residual import (
        automatic_temperature,
        balanced_class_weights,
        class_balanced_transition_rates,
        true_competitor_margin,
    )
    from .train_dcase_stage import macro_accuracy, set_seed
except ImportError:
    from boundary_residual import fix_softmax_gauge
    from train_cached_crossing_residual import (
        automatic_temperature,
        balanced_class_weights,
        class_balanced_transition_rates,
        true_competitor_margin,
    )
    from train_dcase_stage import macro_accuracy, set_seed


class RankGrowingBoundaryResidual(nn.Module):
    def __init__(
        self,
        source_weight: torch.Tensor,
        source_bias: torch.Tensor,
        max_rank: int,
        alpha: float,
    ):
        super().__init__()
        self.register_buffer("source_weight", source_weight.detach().clone())
        self.register_buffer("source_bias", source_bias.detach().clone())
        self.A = nn.Parameter(torch.zeros(max_rank, source_weight.shape[1]))
        self.B = nn.Parameter(torch.zeros(source_weight.shape[0], max_rank))
        self.delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        self.max_rank = int(max_rank)
        self.active_rank = 0
        self.scaling = float(alpha) / max_rank

    def delta_weight(self) -> torch.Tensor:
        if self.active_rank == 0:
            return torch.zeros_like(self.source_weight)
        return self.scaling * (
            self.B[:, : self.active_rank] @ self.A[: self.active_rank]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(
            features,
            self.source_weight + self.delta_weight(),
            self.source_bias + self.delta_bias,
        )

    def grow_from_gradient(
        self, gradient: torch.Tensor, magnitude: float = 0.01
    ) -> None:
        if self.active_rank >= self.max_rank:
            raise RuntimeError("maximum rank already active")
        centered = fix_softmax_gauge(gradient.detach())
        left, _, right_h = torch.linalg.svd(centered, full_matrices=False)
        index = self.active_rank
        with torch.no_grad():
            self.A[index].copy_(right_h[0])
            self.B[:, index].copy_(
                -float(magnitude) * left[:, 0] / self.scaling
            )
        self.active_rank += 1


def class_balanced_mean(
    values: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    weights = class_weights[labels]
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def smooth_harm(
    source_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Straight-through class-balanced harmful-crossing rate.

    The forward value is the actual source-correct to adapted-wrong indicator,
    while its backward pass uses a temperature-scaled logistic relaxation.  This
    gives the dual variable an optimizable signal without replacing the constraint
    by a confidence average that can hide harmful crossings behind safer samples.
    """
    source_margin = true_competitor_margin(source_logits, labels)
    adapted_margin = true_competitor_margin(adapted_logits, labels)
    source_correct = (source_margin > 0).to(adapted_margin.dtype).detach()
    adapted_wrong_soft = torch.sigmoid(
        -adapted_margin / max(temperature, 1e-6)
    )
    adapted_wrong_hard = (adapted_margin <= 0).to(adapted_margin.dtype)
    adapted_wrong = (
        adapted_wrong_hard.detach()
        - adapted_wrong_soft.detach()
        + adapted_wrong_soft
    )
    return class_balanced_mean(
        source_correct * adapted_wrong,
        labels,
        class_weights,
    )


def evaluate(
    model: RankGrowingBoundaryResidual,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        source_logits = F.linear(features, model.source_weight, model.source_bias)
        adapted_logits = model(features)
        smooth = smooth_harm(
            source_logits,
            adapted_logits,
            labels,
            class_weights,
            temperature,
        )
    correcting, harmful, net = class_balanced_transition_rates(
        labels,
        source_logits.argmax(dim=1),
        adapted_logits.argmax(dim=1),
        adapted_logits.shape[1],
    )
    return {
        "validation_loss": float(
            F.cross_entropy(adapted_logits, labels, weight=class_weights)
        ),
        "validation_macro_accuracy": macro_accuracy(
            labels.tolist(), adapted_logits.argmax(dim=1).tolist()
        ),
        "smooth_harm": float(smooth),
        "correcting_crossing_rate": correcting,
        "harmful_crossing_rate": harmful,
        "net_correcting_rate": net,
    }


def metrics_from_logits(
    source_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    correcting, harmful, net = class_balanced_transition_rates(
        labels,
        source_logits.argmax(dim=1),
        adapted_logits.argmax(dim=1),
        adapted_logits.shape[1],
    )
    predictions = adapted_logits.argmax(dim=1)
    class_accuracies = []
    for class_index in range(adapted_logits.shape[1]):
        mask = labels == class_index
        if bool(mask.any()):
            class_accuracies.append(
                float((predictions[mask] == labels[mask]).float().mean())
            )
    return {
        "validation_loss": float(
            F.cross_entropy(adapted_logits, labels, weight=class_weights)
        ),
        "validation_macro_accuracy": 100.0 * float(np.mean(class_accuracies)),
        "smooth_harm": harmful / 100.0,
        "correcting_crossing_rate": correcting,
        "harmful_crossing_rate": harmful,
        "net_correcting_rate": net,
    }


def radial_feasibility_restore(
    model: RankGrowingBoundaryResidual,
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    fit_class_weights: torch.Tensor,
    validation_class_weights: torch.Tensor,
    temperature: float,
    fit_harm_budget: float,
    validation_harm_budget: float,
    grid_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Select a feasible source-anchored scale without increasing the rank.

    Scaling B and delta_bias by rho maps the learned classifier onto the exact
    radial path W(rho)=W0+rho*DeltaW.  rho=0 is the source classifier and is
    always feasible, so this restoration has a non-empty feasible set.
    """
    if grid_size < 2:
        raise ValueError("radial grid must contain at least two points")
    raw_state = copy.deepcopy(model.state_dict())
    with torch.inference_mode():
        source_fit = F.linear(
            fit_features, model.source_weight, model.source_bias
        )
        source_validation = F.linear(
            validation_features, model.source_weight, model.source_bias
        )
        raw_fit = model(fit_features)
        raw_validation = model(validation_features)
    best: tuple[tuple[float, ...], float, dict[str, float]] | None = None
    for radial_scale in torch.linspace(0.0, 1.0, grid_size).tolist():
        adapted_fit = source_fit + radial_scale * (raw_fit - source_fit)
        adapted_validation = source_validation + radial_scale * (
            raw_validation - source_validation
        )
        fit_metrics = metrics_from_logits(
            source_fit,
            adapted_fit,
            fit_labels,
            fit_class_weights,
            temperature,
        )
        validation_metrics = metrics_from_logits(
            source_validation,
            adapted_validation,
            validation_labels,
            validation_class_weights,
            temperature,
        )
        fit_harm = fit_metrics["harmful_crossing_rate"] / 100.0
        validation_harm = (
            validation_metrics["harmful_crossing_rate"] / 100.0
        )
        if fit_harm > fit_harm_budget + 1e-12:
            continue
        if validation_harm > validation_harm_budget + 1e-12:
            continue
        combined = {
            **validation_metrics,
            "fit_harmful_crossing_rate": fit_metrics[
                "harmful_crossing_rate"
            ],
            "fit_correcting_crossing_rate": fit_metrics[
                "correcting_crossing_rate"
            ],
            "fit_net_correcting_rate": fit_metrics["net_correcting_rate"],
            "fit_constraint_violation": max(
                0.0, fit_harm - fit_harm_budget
            ),
            "validation_constraint_violation": max(
                0.0, validation_harm - validation_harm_budget
            ),
            "constraint_violation": 0.0,
            "radial_scale": float(radial_scale),
        }
        score = (
            validation_metrics["validation_macro_accuracy"],
            validation_metrics["net_correcting_rate"],
            -validation_metrics["harmful_crossing_rate"],
            -validation_metrics["validation_loss"],
            -float(radial_scale),
        )
        if best is None or score > best[0]:
            best = score, float(radial_scale), combined
    if best is None:
        raise RuntimeError("radial restoration lost the guaranteed source anchor")
    _, radial_scale, metrics = best
    restored = copy.deepcopy(raw_state)
    restored["B"] = restored["B"] * radial_scale
    restored["delta_bias"] = restored["delta_bias"] * radial_scale
    return restored, metrics


def lagrangian_gradient(
    model: RankGrowingBoundaryResidual,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float,
    dual_value: float,
    epsilon: float,
) -> torch.Tensor:
    delta = model.delta_weight().detach().requires_grad_(True)
    source_logits = F.linear(features, model.source_weight, model.source_bias)
    adapted_logits = F.linear(
        features,
        model.source_weight + delta,
        model.source_bias + model.delta_bias.detach(),
    )
    loss = F.cross_entropy(adapted_logits, labels, weight=class_weights)
    if dual_value > 0:
        loss = loss + dual_value * (
            smooth_harm(
                source_logits,
                adapted_logits,
                labels,
                class_weights,
                temperature,
            )
            - epsilon
        )
    return torch.autograd.grad(loss, delta)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--constraint-mode", choices=("harm", "none"), default="harm")
    parser.add_argument("--max-rank", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--dual-learning-rate", type=float, default=2.0)
    parser.add_argument("--dual-max", type=float, default=50.0)
    parser.add_argument(
        "--harm-slack",
        type=float,
        default=0.005,
        help="Allowed class-balanced harmful-crossing rate on fit data (fraction).",
    )
    parser.add_argument("--growth-magnitude", type=float, default=0.01)
    parser.add_argument("--epochs-per-rank", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selection-tolerance", type=float, default=0.5)
    parser.add_argument("--feasibility-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--feasibility-restoration",
        choices=("radial", "none"),
        default="radial",
    )
    parser.add_argument("--radial-grid-size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=1193)
    args = parser.parse_args()

    set_seed(args.seed)
    cache = np.load(args.cache, allow_pickle=False)
    fit_features = torch.from_numpy(cache["fit_features"].astype(np.float32))
    fit_labels = torch.from_numpy(cache["fit_labels"].astype(np.int64))
    validation_features = torch.from_numpy(
        cache["validation_features"].astype(np.float32)
    )
    validation_labels = torch.from_numpy(
        cache["validation_labels"].astype(np.int64)
    )
    source_weight = torch.from_numpy(cache["source_weight"].astype(np.float32))
    source_bias = torch.from_numpy(cache["source_bias"].astype(np.float32))
    classes = source_weight.shape[0]
    max_rank = min(args.max_rank, classes - 1, source_weight.shape[1])
    if max_rank < 1:
        raise RuntimeError("decision-boundary residual needs at least two classes")
    class_weights = balanced_class_weights(fit_labels, classes)
    validation_class_weights = balanced_class_weights(
        validation_labels, classes
    )
    temperature = automatic_temperature(
        fit_features,
        fit_labels,
        source_weight,
        source_bias,
    )
    with torch.inference_mode():
        source_fit = F.linear(fit_features, source_weight, source_bias)
        baseline_harm = float(
            smooth_harm(
                source_fit,
                source_fit,
                fit_labels,
                class_weights,
                temperature,
            )
        )
    epsilon = baseline_harm + args.harm_slack

    model = RankGrowingBoundaryResidual(
        source_weight,
        source_bias,
        max_rank=max_rank,
        alpha=args.alpha,
    )
    rows = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    raw_states: dict[int, dict[str, torch.Tensor]] = {}
    dual_value = 0.0
    loader = DataLoader(
        TensorDataset(fit_features, fit_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.constraint_mode == "harm":
        source_validation_logits = F.linear(
            validation_features, source_weight, source_bias
        )
        baseline_metrics = metrics_from_logits(
            source_validation_logits,
            source_validation_logits,
            validation_labels,
            validation_class_weights,
            temperature,
        )
        baseline_row = {
            "rank": 0,
            "best_epoch": 0,
            "dual_value": 0.0,
            "epsilon": epsilon,
            "baseline_smooth_harm": baseline_harm,
            "constraint_violation": 0.0,
            "fit_constraint_violation": 0.0,
            "validation_constraint_violation": 0.0,
            "fit_harmful_crossing_rate": 0.0,
            "fit_correcting_crossing_rate": 0.0,
            "fit_net_correcting_rate": 0.0,
            "radial_scale": 0.0,
            "stored_floats": 0,
            **baseline_metrics,
        }
        rows.append(baseline_row)
        states[0] = copy.deepcopy(model.state_dict())
        raw_states[0] = copy.deepcopy(model.state_dict())

    def restored_candidate() -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, float],
    ]:
        raw_state = copy.deepcopy(model.state_dict())
        if (
            args.constraint_mode == "harm"
            and args.feasibility_restoration == "radial"
        ):
            restored_state, metrics = radial_feasibility_restore(
                model,
                fit_features,
                fit_labels,
                validation_features,
                validation_labels,
                class_weights,
                validation_class_weights,
                temperature,
                epsilon,
                epsilon + args.feasibility_tolerance,
                args.radial_grid_size,
            )
            return raw_state, restored_state, metrics
        state = copy.deepcopy(raw_state)
        metrics = evaluate(
            model,
            validation_features,
            validation_labels,
            validation_class_weights,
            temperature,
        )
        metrics.update(
            {
                "fit_harmful_crossing_rate": float("nan"),
                "fit_correcting_crossing_rate": float("nan"),
                "fit_net_correcting_rate": float("nan"),
                "fit_constraint_violation": float("nan"),
                "validation_constraint_violation": max(
                    0.0, metrics["smooth_harm"] - epsilon
                ),
                "constraint_violation": max(
                    0.0, metrics["smooth_harm"] - epsilon
                ),
                "radial_scale": 1.0,
            }
        )
        return raw_state, state, metrics

    for rank in range(1, max_rank + 1):
        gradient = lagrangian_gradient(
            model,
            fit_features,
            fit_labels,
            class_weights,
            temperature,
            dual_value if args.constraint_mode == "harm" else 0.0,
            epsilon,
        )
        model.grow_from_gradient(gradient, args.growth_magnitude)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_raw_state, best_state, best_metrics = restored_candidate()
        best_epoch = 0
        stale = 0
        for epoch in range(1, args.epochs_per_rank + 1):
            model.train()
            for features, labels in loader:
                source_logits = F.linear(
                    features, model.source_weight, model.source_bias
                ).detach()
                adapted_logits = model(features)
                loss = F.cross_entropy(
                    adapted_logits,
                    labels,
                    weight=class_weights,
                )
                if args.constraint_mode == "harm":
                    loss = loss + dual_value * (
                        smooth_harm(
                            source_logits,
                            adapted_logits,
                            labels,
                            class_weights,
                            temperature,
                        )
                        - epsilon
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Inactive rows/columns are outside the forward graph, but masking
                # makes that invariance explicit for future optimizer changes.
                if model.A.grad is not None:
                    model.A.grad[model.active_rank :] = 0
                if model.B.grad is not None:
                    model.B.grad[:, model.active_rank :] = 0
                optimizer.step()

            if args.constraint_mode == "harm":
                with torch.inference_mode():
                    adapted_fit = model(fit_features)
                    epoch_harm = float(
                        smooth_harm(
                            source_fit,
                            adapted_fit,
                            fit_labels,
                            class_weights,
                            temperature,
                        )
                    )
                dual_value = min(
                    args.dual_max,
                    max(
                        0.0,
                        dual_value
                        + args.dual_learning_rate * (epoch_harm - epsilon),
                    ),
                )
            candidate_raw_state, candidate_state, metrics = restored_candidate()
            def constrained_score(values: dict[str, float]) -> tuple[float, ...]:
                violation = max(0.0, values["smooth_harm"] - epsilon)
                feasible = violation <= args.feasibility_tolerance
                # Accuracy is optimized only inside the feasible set.  If the
                # constraint is violated, restoration of feasibility has priority.
                return (
                    float(feasible),
                    values["validation_macro_accuracy"] if feasible else -violation,
                    -values["harmful_crossing_rate"],
                    -values["validation_loss"],
                )

            candidate = constrained_score(metrics)
            incumbent = constrained_score(best_metrics)
            if candidate > incumbent:
                best_raw_state = candidate_raw_state
                best_state = candidate_state
                best_metrics = metrics
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        row = {
            "rank": rank,
            "best_epoch": best_epoch,
            "dual_value": dual_value,
            "epsilon": epsilon,
            "baseline_smooth_harm": baseline_harm,
            "stored_floats": rank * (classes + source_weight.shape[1]) + classes,
            **best_metrics,
        }
        rows.append(row)
        raw_states[rank] = best_raw_state
        states[rank] = best_state
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.constraint_mode == "harm":
        feasible = [row for row in rows if row["constraint_violation"] <= 1e-12]
        pool = feasible or rows
    else:
        pool = rows
    best_accuracy = max(row["validation_macro_accuracy"] for row in pool)
    eligible = [
        row
        for row in pool
        if row["validation_macro_accuracy"]
        >= best_accuracy - args.selection_tolerance
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["rank"],
            row["constraint_violation"],
            -row["validation_macro_accuracy"],
        ),
    )
    selected_rank = int(selected["rank"])
    selected_state = states[selected_rank]
    selected_model = RankGrowingBoundaryResidual(
        source_weight,
        source_bias,
        max_rank=max_rank,
        alpha=args.alpha,
    )
    selected_model.active_rank = selected_rank
    selected_model.load_state_dict(selected_state)
    delta_weight = fix_softmax_gauge(selected_model.delta_weight().detach())
    restoration_fields = (
        "rank",
        "best_epoch",
        "dual_value",
        "epsilon",
        "baseline_smooth_harm",
        "radial_scale",
        "validation_loss",
        "validation_macro_accuracy",
        "harmful_crossing_rate",
        "correcting_crossing_rate",
        "net_correcting_rate",
        "fit_harmful_crossing_rate",
        "fit_correcting_crossing_rate",
        "fit_net_correcting_rate",
        "fit_constraint_violation",
        "validation_constraint_violation",
        "constraint_violation",
    )
    rank_restoration = {
        int(row["rank"]): {
            key: row[key] for key in restoration_fields if key in row
        }
        for row in rows
    }
    payload = {
        "format": "icassp2027-hcrg-boundary-v1",
        "constraint_mode": args.constraint_mode,
        "selected_rank": selected_rank,
        "requested_max_rank": int(args.max_rank),
        "max_rank": max_rank,
        "alpha": float(args.alpha),
        "factor_scaling": float(args.alpha) / max_rank,
        "candidate_ranks": sorted(states),
        "candidate_raw_states": raw_states,
        "candidate_restored_states": states,
        "rank_restoration": rank_restoration,
        "state_semantics": {
            "candidate_raw_states": (
                "unscaled state at the winning epoch before radial feasibility "
                "restoration; use nonzero ranks as exact-rho path endpoints"
            ),
            "candidate_restored_states": (
                "the same winning states after B and delta_bias are multiplied "
                "by rank_restoration[rank].radial_scale; these states preserve "
                "the original HCRG continuation trajectory"
            ),
            "selected_state_dict": "selected restored state (legacy behavior)",
        },
        "selected_state_dict": selected_state,
        "delta_weight": delta_weight,
        "delta_bias": selected_model.delta_bias.detach().clone(),
        "source_weight": source_weight,
        "source_bias": source_bias,
        "rows": rows,
        "epsilon": epsilon,
        "baseline_smooth_harm": baseline_harm,
        "temperature": temperature,
        "harm_slack": args.harm_slack,
        "dual_learning_rate": args.dual_learning_rate,
        "dual_max": args.dual_max,
        "selection_tolerance": args.selection_tolerance,
        "feasibility_tolerance": args.feasibility_tolerance,
        "feasibility_restoration": args.feasibility_restoration,
        "radial_grid_size": args.radial_grid_size,
        "selected_radial_scale": float(selected.get("radial_scale", 1.0)),
        "seed": args.seed,
        "stage": int(cache["stage"]),
        "task": int(cache["task"]),
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    frame = pd.DataFrame(rows)
    frame["selected"] = frame["rank"] == selected_rank
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary, index=False)
    print(json.dumps({"selected": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
