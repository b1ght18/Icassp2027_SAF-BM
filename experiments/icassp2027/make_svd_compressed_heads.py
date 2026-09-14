from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .boundary_residual import FixedRankBoundaryResidual, fix_softmax_gauge
    from .train_dcase_stage import macro_accuracy
except ImportError:
    from boundary_residual import FixedRankBoundaryResidual, fix_softmax_gauge
    from train_dcase_stage import macro_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc SVD compression of a validation-trained full boundary update"
    )
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ranks", default="1,2,4,8,9")
    parser.add_argument("--full-rank", type=int, default=10)
    parser.add_argument("--selection-tolerance", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    args = parser.parse_args()

    source_payload = torch.load(
        args.head_checkpoint, map_location="cpu", weights_only=False
    )
    states = source_payload["all_states"]
    full_state = states.get(args.full_rank, states.get(str(args.full_rank)))
    if full_state is None:
        raise RuntimeError(f"Rank {args.full_rank} is unavailable")
    full = FixedRankBoundaryResidual(
        full_state["source_weight"],
        full_state["source_bias"],
        args.full_rank,
        args.alpha,
    )
    full.load_state_dict(full_state, strict=True)
    delta = full.delta_weight().detach()
    delta_bias = full.delta_bias.detach()
    # Remove the softmax-common logit gauge. This changes every class logit by
    # the same sample-dependent scalar and therefore preserves CE and argmax.
    delta, delta_bias = fix_softmax_gauge(delta, delta_bias)
    left, singular, right = torch.linalg.svd(delta, full_matrices=False)

    cache = np.load(args.cache, allow_pickle=False)
    validation_features = torch.from_numpy(
        cache["validation_features"].astype(np.float32)
    )
    validation_labels = torch.from_numpy(cache["validation_labels"].astype(np.int64))
    source_weight = full.source_weight.detach()
    source_bias = full.source_bias.detach()
    output_states: dict[int, dict[str, torch.Tensor]] = {}
    rows = []
    total_energy = singular.square().sum()
    for rank in [int(value) for value in args.ranks.split(",")]:
        maximum_boundary_rank = source_weight.shape[0] - 1
        if not 1 <= rank <= maximum_boundary_rank:
            raise ValueError(f"Rank {rank} is outside [1, {maximum_boundary_rank}]")
        truncated = (left[:, :rank] * singular[:rank].unsqueeze(0)) @ right[:rank]
        # Encode the already-known truncated SVD directly.  Re-running SVD in
        # ``to_fixed_state`` can promote float32 reconstruction noise just
        # above its numerical-rank tolerance (notably for four-class heads).
        model = FixedRankBoundaryResidual(
            source_weight, source_bias, rank, args.alpha
        )
        scaling = args.alpha / rank
        root = torch.sqrt(singular[:rank] / scaling)
        with torch.no_grad():
            model.B.copy_(left[:, :rank] * root.unsqueeze(0))
            model.A.copy_(root.unsqueeze(1) * right[:rank])
            model.delta_bias.copy_(delta_bias)
        if not torch.allclose(
            model.delta_weight(), truncated, atol=2e-5, rtol=2e-5
        ):
            raise RuntimeError("direct SVD factor encoding changed the truncation")
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        output_states[rank] = state
        with torch.inference_mode():
            logits = F.linear(
                validation_features, source_weight + truncated, source_bias + delta_bias
            )
            loss = float(F.cross_entropy(logits, validation_labels))
            accuracy = macro_accuracy(
                validation_labels.tolist(), logits.argmax(1).tolist()
            )
        squared = singular[:rank].square()
        rows.append(
            {
                "rank": rank,
                "validation_macro_accuracy": accuracy,
                "validation_loss": loss,
                "retained_full_update_energy": float(squared.sum() / total_energy),
                "stable_rank": float(squared.sum() / squared.max().clamp_min(1e-20)),
                "stored_floats": rank * (source_weight.shape[0] + source_weight.shape[1])
                + source_weight.shape[0],
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    best_accuracy = max(row["validation_macro_accuracy"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["validation_macro_accuracy"]
        >= best_accuracy - args.selection_tolerance
    ]
    selected = min(eligible, key=lambda row: row["rank"])
    selected_rank = int(selected["rank"])
    output = {
        "format": "icassp2027-posthoc-svd-boundary-rank-v1",
        "selected_rank": selected_rank,
        "selected_state_dict": output_states[selected_rank],
        "all_states": output_states,
        "rows": rows,
        "selection_tolerance": args.selection_tolerance,
        "alpha": args.alpha,
        "full_rank": args.full_rank,
        "source_head_checkpoint": str(args.head_checkpoint.resolve()),
        "task": int(source_payload["task"]),
        "stage": int(source_payload["stage"]),
        "seed": int(source_payload["seed"]),
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    frame = pd.DataFrame(rows)
    frame["selected"] = frame["rank"] == selected_rank
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary, index=False)
    print(
        json.dumps(
            {"best_validation": best_accuracy, "selected_rank": selected_rank},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
