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


def class_weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    weights = np.zeros(classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights[present] *= present.sum() / weights[present].sum()
    return torch.from_numpy(weights)


def evaluate(model, features: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
    return float(F.cross_entropy(logits, labels)), macro_accuracy(labels.tolist(), logits.argmax(1).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-adaptive direct boundary rank selection")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ranks", default="1,2,4,8,10")
    parser.add_argument("--learning-rates", default="0.001,0.003,0.01")
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selection-tolerance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1193)
    args = parser.parse_args()
    data = np.load(args.cache, allow_pickle=False)
    fit_features = torch.from_numpy(data["fit_features"].astype(np.float32))
    fit_labels = torch.from_numpy(data["fit_labels"].astype(np.int64))
    validation_features = torch.from_numpy(data["validation_features"].astype(np.float32))
    validation_labels = torch.from_numpy(data["validation_labels"].astype(np.int64))
    source_weight = torch.from_numpy(data["source_weight"].astype(np.float32))
    source_bias = torch.from_numpy(data["source_bias"].astype(np.float32))
    classes = source_weight.shape[0]
    weights = class_weights(data["fit_labels"], classes)
    rows, states = [], {}

    for rank in [int(value) for value in args.ranks.split(",")]:
        best_rank = None
        for learning_rate in [float(value) for value in args.learning_rates.split(",")]:
            set_seed(args.seed)
            model = FixedRankBoundaryResidual(source_weight, source_bias, rank, args.alpha)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            dataset = TensorDataset(fit_features, fit_labels)
            generator = torch.Generator().manual_seed(args.seed)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
            best_state, best_accuracy, best_loss, best_epoch, stale = None, -math.inf, math.inf, 0, 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                for features, labels in loader:
                    loss = F.cross_entropy(model(features), labels, weight=weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                validation_loss, accuracy = evaluate(model, validation_features, validation_labels)
                if accuracy > best_accuracy + 1e-9:
                    best_state = copy.deepcopy(model.state_dict())
                    best_accuracy, best_loss, best_epoch, stale = accuracy, validation_loss, epoch, 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            candidate = (best_accuracy, -best_epoch, learning_rate)
            if best_rank is None or candidate > best_rank[0]:
                best_rank = (candidate, best_state, best_loss, best_epoch, learning_rate)
        assert best_rank is not None
        model = FixedRankBoundaryResidual(source_weight, source_bias, rank, args.alpha)
        model.load_state_dict(best_rank[1])
        delta = fix_softmax_gauge(model.delta_weight().detach())
        singular = torch.linalg.svdvals(delta)
        squared = singular.square()
        numerical_rank = int(
            (singular > singular.max().clamp_min(1e-20) * 1e-6).sum().item()
        )
        row = {
            "rank": rank,
            "validation_macro_accuracy": best_rank[0][0],
            "validation_loss": best_rank[2],
            "best_epoch": best_rank[3],
            "best_learning_rate": best_rank[4],
            "stable_rank": float(squared.sum() / squared.max().clamp_min(1e-20)),
            "numerical_boundary_rank": numerical_rank,
            "gauge_fixed_stored_floats": numerical_rank
            * (source_weight.shape[0] + source_weight.shape[1])
            + classes,
            "stored_floats": rank * (source_weight.shape[0] + source_weight.shape[1]) + classes,
        }
        rows.append(row)
        states[rank] = best_rank[1]
        print(json.dumps(row, sort_keys=True), flush=True)

    best_accuracy = max(row["validation_macro_accuracy"] for row in rows)
    eligible = [
        row for row in rows
        if row["validation_macro_accuracy"] >= best_accuracy - args.selection_tolerance
    ]
    selected = min(eligible, key=lambda row: row["rank"])
    payload = {
        "format": "icassp2027-validation-adaptive-boundary-rank-v1",
        "selected_rank": selected["rank"],
        "selected_state_dict": states[selected["rank"]],
        "all_states": states,
        "rows": rows,
        "selection_tolerance": args.selection_tolerance,
        "candidate_ranks": [int(value) for value in args.ranks.split(",")],
        "learning_rates": [float(value) for value in args.learning_rates.split(",")],
        "alpha": args.alpha,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "loss": "inverse-frequency-class-balanced-cross-entropy",
        "task": int(data["task"]),
        "stage": int(data["stage"]),
        "seed": args.seed,
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    frame = pd.DataFrame(rows)
    frame["selected"] = frame["rank"] == selected["rank"]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary, index=False)
    print(json.dumps({"selected": selected, "best_validation": best_accuracy}, sort_keys=True))


if __name__ == "__main__":
    main()
