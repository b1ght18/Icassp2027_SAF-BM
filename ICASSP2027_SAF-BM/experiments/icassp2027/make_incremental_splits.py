from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .manifest import load_manifest, write_manifest
except ImportError:
    from manifest import load_manifest, write_manifest


def assign_grouped_validation(
    domain_frame: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> pd.Series:
    assignment = pd.Series("fit", index=domain_frame.index, dtype="string")
    generator = np.random.default_rng(seed)
    for label, label_frame in domain_frame.groupby("label", sort=True):
        groups = label_frame.recording_group.drop_duplicates().to_numpy()
        generator.shuffle(groups)
        validation_count = max(1, int(round(validation_fraction * len(groups))))
        if validation_count >= len(groups):
            validation_count = max(0, len(groups) - 1)
        validation_groups = set(groups[:validation_count])
        assignment.loc[label_frame.index[label_frame.recording_group.isin(validation_groups)]] = "validation"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")

    frame = load_manifest(args.manifest)
    frame = frame.copy()
    frame.loc[frame.official_partition == "test", "usage"] = "test"
    frame.loc[frame.official_partition == "evaluation", "usage"] = "evaluation"
    training = frame.official_partition == "train"
    for domain_index, domain_frame in frame[training].groupby("domain_index", sort=True):
        assignment = assign_grouped_validation(
            domain_frame,
            args.validation_fraction,
            args.seed + 1009 * int(domain_index),
        )
        frame.loc[domain_frame.index, "usage"] = assignment
    frame["split_seed"] = args.seed
    frame["validation_fraction"] = args.validation_fraction
    write_manifest(frame, args.output)
    print(frame.groupby(["domain", "usage"], dropna=False).size().to_string())
    print(f"Saved split manifest to {args.output}")


if __name__ == "__main__":
    main()
