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
from torch.utils.data import DataLoader, TensorDataset

try:
    from .boundary_residual import FixedRankBoundaryResidual, fix_softmax_gauge
    from .train_dcase_stage import macro_accuracy, set_seed
except ImportError:
    from boundary_residual import FixedRankBoundaryResidual, fix_softmax_gauge
    from train_dcase_stage import macro_accuracy, set_seed


def balanced_class_weights(labels: torch.Tensor, classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=classes).float()
    weights = torch.zeros(classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights[present] *= present.sum() / weights[present].sum()
    return weights


def true_competitor_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(),
        -torch.inf,
    )
    return true_logits - competitors.max(dim=1).values


def crossing_surrogate(
    source_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth harmful-minus-correcting boundary-crossing objective.

    A positive true-vs-best-competitor margin denotes a correct decision. The
    source-normalized classifier supplies the reference boundary state. The loss
    rewards negative-to-positive crossings and penalizes positive-to-negative
    crossings without treating arbitrary logit displacement as useful migration.
    """

    source_margin = true_competitor_margin(source_logits, labels)
    adapted_margin = true_competitor_margin(adapted_logits, labels)
    scale = max(float(temperature), 1e-6)
    source_wrong = torch.sigmoid(-source_margin / scale).detach()
    source_right = torch.sigmoid(source_margin / scale).detach()
    adapted_right = torch.sigmoid(adapted_margin / scale)
    adapted_wrong = torch.sigmoid(-adapted_margin / scale)
    correcting = (source_wrong * adapted_right).mean()
    harmful = (source_right * adapted_wrong).mean()
    return harmful - correcting, correcting, harmful


def class_balanced_transition_rates(
    labels: torch.Tensor,
    source_predictions: torch.Tensor,
    adapted_predictions: torch.Tensor,
    classes: int,
) -> tuple[float, float, float]:
    correcting, harmful = [], []
    for class_index in range(classes):
        mask = labels == class_index
        if not bool(mask.any()):
            continue
        source_correct = source_predictions[mask] == labels[mask]
        adapted_correct = adapted_predictions[mask] == labels[mask]
        correcting.append(float((~source_correct & adapted_correct).float().mean()))
        harmful.append(float((source_correct & ~adapted_correct).float().mean()))
    correcting_rate = 100.0 * float(np.mean(correcting)) if correcting else 0.0
    harmful_rate = 100.0 * float(np.mean(harmful)) if harmful else 0.0
    return correcting_rate, harmful_rate, correcting_rate - harmful_rate


def evaluate(
    model: FixedRankBoundaryResidual,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        source_logits = F.linear(features, model.source_weight, model.source_bias)
        adapted_logits = model(features)
    correcting, harmful, net = class_balanced_transition_rates(
        labels,
        source_logits.argmax(dim=1),
        adapted_logits.argmax(dim=1),
        adapted_logits.shape[1],
    )
    return {
        "validation_loss": float(F.cross_entropy(adapted_logits, labels)),
        "validation_macro_accuracy": macro_accuracy(
            labels.tolist(), adapted_logits.argmax(dim=1).tolist()
        ),
        "correcting_crossing_rate": correcting,
        "harmful_crossing_rate": harmful,
        "net_correcting_rate": net,
    }


def automatic_temperature(
    features: torch.Tensor,
    labels: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
) -> float:
    with torch.inference_mode():
        logits = F.linear(features, source_weight, source_bias)
        margins = true_competitor_margin(logits, labels).abs()
    return max(float(torch.quantile(margins, 0.5)), 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot crossing-aware optimization of cached boundary residuals"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ranks", default="1,2")
    parser.add_argument("--learning-rates", default="0.003,0.01")
    parser.add_argument("--crossing-weights", default="0,0.1,0.3,1")
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selection-tolerance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1193)
    args = parser.parse_args()

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
    class_weights = balanced_class_weights(fit_labels, classes)
    temperature = automatic_temperature(
        fit_features, fit_labels, source_weight, source_bias
    )

    rows: list[dict[str, float | int]] = []
    states: dict[tuple[int, float], dict[str, torch.Tensor]] = {}
    dataset = TensorDataset(fit_features, fit_labels)
    ranks = [int(value) for value in args.ranks.split(",")]
    learning_rates = [float(value) for value in args.learning_rates.split(",")]
    crossing_weights = [float(value) for value in args.crossing_weights.split(",")]

    for rank in ranks:
        for crossing_weight in crossing_weights:
            best: tuple[
                tuple[float, float, float, int],
                dict[str, torch.Tensor],
                dict[str, float],
                float,
                int,
            ] | None = None
            for learning_rate in learning_rates:
                set_seed(args.seed)
                model = FixedRankBoundaryResidual(
                    source_weight, source_bias, rank, args.alpha
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                generator = torch.Generator().manual_seed(args.seed)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    generator=generator,
                )
                best_state, best_metrics = None, None
                best_epoch, stale = 0, 0
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    for features, labels in loader:
                        source_logits = F.linear(
                            features, model.source_weight, model.source_bias
                        ).detach()
                        adapted_logits = model(features)
                        ce = F.cross_entropy(
                            adapted_logits, labels, weight=class_weights
                        )
                        crossing, _, _ = crossing_surrogate(
                            source_logits,
                            adapted_logits,
                            labels,
                            temperature,
                        )
                        loss = ce + crossing_weight * crossing
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                    metrics = evaluate(
                        model, validation_features, validation_labels
                    )
                    candidate = (
                        metrics["validation_macro_accuracy"],
                        metrics["net_correcting_rate"],
                        -metrics["validation_loss"],
                        -epoch,
                    )
                    incumbent = (
                        -math.inf,
                        -math.inf,
                        -math.inf,
                        -math.inf,
                    )
                    if best_metrics is not None:
                        incumbent = (
                            best_metrics["validation_macro_accuracy"],
                            best_metrics["net_correcting_rate"],
                            -best_metrics["validation_loss"],
                            -best_epoch,
                        )
                    if candidate > incumbent:
                        best_state = copy.deepcopy(model.state_dict())
                        best_metrics = metrics
                        best_epoch, stale = epoch, 0
                    else:
                        stale += 1
                        if stale >= args.patience:
                            break

                assert best_state is not None and best_metrics is not None
                candidate = (
                    best_metrics["validation_macro_accuracy"],
                    best_metrics["net_correcting_rate"],
                    -best_metrics["validation_loss"],
                    -best_epoch,
                )
                if best is None or candidate > best[0]:
                    best = (
                        candidate,
                        best_state,
                        best_metrics,
                        learning_rate,
                        best_epoch,
                    )

            assert best is not None
            model = FixedRankBoundaryResidual(
                source_weight, source_bias, rank, args.alpha
            )
            model.load_state_dict(best[1])
            delta = fix_softmax_gauge(model.delta_weight().detach())
            singular = torch.linalg.svdvals(delta)
            squared = singular.square()
            numerical_rank = int(
                (singular > singular.max().clamp_min(1e-20) * 1e-6).sum().item()
            )
            row: dict[str, float | int] = {
                "rank": rank,
                "crossing_weight": crossing_weight,
                "crossing_temperature": temperature,
                "best_learning_rate": best[3],
                "best_epoch": best[4],
                **best[2],
                "stable_rank": float(
                    squared.sum() / squared.max().clamp_min(1e-20)
                ),
                "numerical_boundary_rank": numerical_rank,
                "stored_floats": rank * (classes + source_weight.shape[1]) + classes,
            }
            rows.append(row)
            states[(rank, crossing_weight)] = best[1]
            print(json.dumps(row, sort_keys=True), flush=True)

    best_accuracy = max(float(row["validation_macro_accuracy"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["validation_macro_accuracy"])
        >= best_accuracy - args.selection_tolerance
    ]
    selected = min(
        eligible,
        key=lambda row: (
            int(row["rank"]),
            -float(row["validation_macro_accuracy"]),
            -float(row["net_correcting_rate"]),
        ),
    )
    key = (int(selected["rank"]), float(selected["crossing_weight"]))
    payload = {
        "format": "icassp2027-crossing-aware-boundary-pilot-v1",
        "selected_rank": key[0],
        "selected_crossing_weight": key[1],
        "selected_state_dict": states[key],
        "rows": rows,
        "selection_tolerance": args.selection_tolerance,
        "candidate_ranks": ranks,
        "learning_rates": learning_rates,
        "crossing_weights": crossing_weights,
        "crossing_temperature": temperature,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "loss": "balanced-ce-plus-smooth-harmful-minus-correcting-crossings",
        "task": int(cache["task"]),
        "stage": int(cache["stage"]),
        "seed": args.seed,
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    frame = pd.DataFrame(rows)
    frame["selected"] = (
        (frame["rank"] == key[0])
        & (frame["crossing_weight"] == key[1])
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary, index=False)
    print(json.dumps({"selected": selected, "best_validation": best_accuracy}, sort_keys=True))


if __name__ == "__main__":
    main()
