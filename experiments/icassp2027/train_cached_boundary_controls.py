from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

try:
    from .boundary_residual import FixedRankBoundaryResidual, fix_softmax_gauge
    from .train_cached_nuclear_residual import to_fixed_state
    from .train_dcase_stage import macro_accuracy, set_seed
except ImportError:
    from boundary_residual import FixedRankBoundaryResidual, fix_softmax_gauge
    from train_cached_nuclear_residual import to_fixed_state
    from train_dcase_stage import macro_accuracy, set_seed


def balanced_sample_weights(labels: np.ndarray, classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    result = np.zeros(len(labels), dtype=np.float64)
    for class_index, count in enumerate(counts):
        if count > 0:
            result[labels == class_index] = 1.0 / count
    return result / max(result.mean(), np.finfo(np.float64).eps)


def balanced_class_weights(labels: torch.Tensor, classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=classes).float()
    weights = torch.zeros(classes)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights[present] *= present.sum() / weights[present].sum()
    return weights


def score(
    features: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[float, float]:
    with torch.inference_mode():
        logits = F.linear(features, weight, bias)
    return (
        float(F.cross_entropy(logits, labels)),
        macro_accuracy(labels.tolist(), logits.argmax(1).tolist()),
    )


def fit_ridge(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    alphas: list[float],
    classes: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]], dict[str, float]]:
    x_fit = fit_features.numpy().astype(np.float64, copy=False)
    y_fit = np.eye(classes, dtype=np.float64)[fit_labels.numpy()]
    x_validation = validation_features.numpy().astype(np.float64, copy=False)
    sample_weight = balanced_sample_weights(fit_labels.numpy(), classes)
    rows: list[dict[str, float]] = []
    best: tuple[tuple[float, float], torch.Tensor, torch.Tensor, dict[str, float]] | None = None
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-7)
        model.fit(x_fit, y_fit, sample_weight=sample_weight)
        weight = torch.from_numpy(model.coef_.astype(np.float32, copy=False))
        bias = torch.from_numpy(np.asarray(model.intercept_, dtype=np.float32))
        loss, accuracy = score(validation_features, validation_labels, weight, bias)
        row = {
            "ridge_alpha": float(alpha),
            "validation_loss": loss,
            "validation_macro_accuracy": accuracy,
        }
        rows.append(row)
        candidate = (accuracy, -loss)
        if best is None or candidate > best[0]:
            best = (candidate, weight, bias, row)
    assert best is not None
    return best[1], best[2], rows, best[3]


def fit_full_ce(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    learning_rates: list[float],
    weight_decays: list[float],
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]], dict[str, float]]:
    classes = source_weight.shape[0]
    class_weights = balanced_class_weights(fit_labels, classes)
    rows: list[dict[str, float]] = []
    best_overall: tuple[tuple[float, float, int], dict[str, torch.Tensor], dict[str, float]] | None = None
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            set_seed(seed)
            head = nn.Linear(source_weight.shape[1], classes)
            with torch.no_grad():
                head.weight.copy_(source_weight)
                head.bias.copy_(source_bias)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                TensorDataset(fit_features, fit_labels),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
            best_state: dict[str, torch.Tensor] | None = None
            best_accuracy, best_loss, best_epoch, stale = -math.inf, math.inf, 0, 0
            for epoch in range(1, epochs + 1):
                head.train()
                for features, labels in loader:
                    loss = F.cross_entropy(head(features), labels, weight=class_weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                loss, accuracy = score(
                    validation_features,
                    validation_labels,
                    head.weight,
                    head.bias,
                )
                if (accuracy, -loss) > (best_accuracy, -best_loss):
                    best_state = copy.deepcopy(head.state_dict())
                    best_accuracy, best_loss, best_epoch, stale = accuracy, loss, epoch, 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            assert best_state is not None
            row = {
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "best_epoch": int(best_epoch),
                "validation_loss": float(best_loss),
                "validation_macro_accuracy": float(best_accuracy),
            }
            rows.append(row)
            candidate = (best_accuracy, -best_loss, -best_epoch)
            if best_overall is None or candidate > best_overall[0]:
                best_overall = (candidate, best_state, row)
    assert best_overall is not None
    return (
        best_overall[1]["weight"],
        best_overall[1]["bias"],
        rows,
        best_overall[2],
    )


def fit_bias_only(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    learning_rates: list[float],
    epochs: int,
    patience: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]], dict[str, float]]:
    class_weights = balanced_class_weights(fit_labels, source_weight.shape[0])
    rows: list[dict[str, float]] = []
    best_overall: tuple[tuple[float, float], torch.Tensor, dict[str, float]] | None = None
    for learning_rate in learning_rates:
        delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        optimizer = torch.optim.Adam([delta_bias], lr=learning_rate)
        best_bias, best_accuracy, best_loss, best_epoch, stale = None, -math.inf, math.inf, 0, 0
        for epoch in range(1, epochs + 1):
            logits = F.linear(fit_features, source_weight, source_bias + delta_bias)
            loss = F.cross_entropy(logits, fit_labels, weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            validation_loss, accuracy = score(
                validation_features,
                validation_labels,
                source_weight,
                source_bias + delta_bias,
            )
            if (accuracy, -validation_loss) > (best_accuracy, -best_loss):
                best_bias = delta_bias.detach().clone()
                best_accuracy, best_loss, best_epoch, stale = (
                    accuracy,
                    validation_loss,
                    epoch,
                    0,
                )
            else:
                stale += 1
                if stale >= patience:
                    break
        assert best_bias is not None
        row = {
            "learning_rate": float(learning_rate),
            "best_epoch": int(best_epoch),
            "validation_loss": float(best_loss),
            "validation_macro_accuracy": float(best_accuracy),
        }
        rows.append(row)
        candidate = (best_accuracy, -best_loss)
        if best_overall is None or candidate > best_overall[0]:
            best_overall = (candidate, best_bias, row)
    assert best_overall is not None
    return (
        source_weight.clone(),
        source_bias + best_overall[1],
        rows,
        best_overall[2],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strong decision-boundary controls on frozen stage-local features"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--method", choices=("bn_ridge", "full_ce", "bias_only"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--learning-rates", default="0.0001,0.0003,0.001,0.003")
    parser.add_argument("--weight-decays", default="0,0.00001,0.0001")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1193)
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=False)
    fit_features = torch.from_numpy(cache["fit_features"].astype(np.float32))
    fit_labels = torch.from_numpy(cache["fit_labels"].astype(np.int64))
    validation_features = torch.from_numpy(cache["validation_features"].astype(np.float32))
    validation_labels = torch.from_numpy(cache["validation_labels"].astype(np.int64))
    source_weight = torch.from_numpy(cache["source_weight"].astype(np.float32))
    source_bias = torch.from_numpy(cache["source_bias"].astype(np.float32))

    if args.method == "bn_ridge":
        effective_weight, effective_bias, rows, selected = fit_ridge(
            fit_features,
            fit_labels,
            validation_features,
            validation_labels,
            [float(value) for value in args.ridge_alphas.split(",")],
            source_weight.shape[0],
        )
    elif args.method == "full_ce":
        effective_weight, effective_bias, rows, selected = fit_full_ce(
            fit_features,
            fit_labels,
            validation_features,
            validation_labels,
            source_weight,
            source_bias,
            [float(value) for value in args.learning_rates.split(",")],
            [float(value) for value in args.weight_decays.split(",")],
            args.epochs,
            args.patience,
            args.batch_size,
            args.seed,
        )
    else:
        effective_weight, effective_bias, rows, selected = fit_bias_only(
            fit_features,
            fit_labels,
            validation_features,
            validation_labels,
            source_weight,
            source_bias,
            [float(value) for value in args.learning_rates.split(",")],
            args.epochs,
            args.patience,
        )

    delta_weight, delta_bias = fix_softmax_gauge(
        effective_weight - source_weight, effective_bias - source_bias
    )
    encoded_rank, state = to_fixed_state(
        source_weight, source_bias, delta_weight, delta_bias, args.alpha
    )
    _, validation_accuracy = score(
        validation_features,
        validation_labels,
        source_weight + delta_weight,
        source_bias + delta_bias,
    )
    payload = {
        "format": "icassp2027-cached-boundary-control-v1",
        "control_method": args.method,
        "selected_rank": encoded_rank,
        "selected_state_dict": state,
        "all_states": {encoded_rank: state},
        "rows": [
            {
                "rank": encoded_rank,
                "validation_macro_accuracy": validation_accuracy,
                "validation_loss": float(selected["validation_loss"]),
                "control_method": args.method,
                "selection": selected,
            }
        ],
        "search_rows": rows,
        "selection_tolerance": 0.0,
        "alpha": args.alpha,
        "task": int(cache["task"]),
        "stage": int(cache["stage"]),
        "seed": args.seed,
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    print(
        json.dumps(
            {
                "method": args.method,
                "stage": payload["stage"],
                "task": payload["task"],
                "encoded_rank": encoded_rank,
                "validation_macro_accuracy": validation_accuracy,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
