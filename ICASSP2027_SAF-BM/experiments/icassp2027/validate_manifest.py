from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .manifest import (
        NON_TRAIN_USAGES,
        REQUIRED_COLUMNS,
        TRAIN_USAGES,
        absolute_paths,
        load_manifest,
    )
except ImportError:
    from manifest import (
        NON_TRAIN_USAGES,
        REQUIRED_COLUMNS,
        TRAIN_USAGES,
        absolute_paths,
        load_manifest,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def intersections(frame: pd.DataFrame, column: str, first: str, second: str) -> set[str]:
    a = set(frame.loc[frame.usage == first, column].astype(str))
    b = set(frame.loc[frame.usage == second, column].astype(str))
    return a & b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--require-files", action="store_true")
    parser.add_argument("--forbid-cross-domain-content-overlap", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    frame = load_manifest(args.manifest)

    require(not frame.sample_id.duplicated().any(), "Duplicate sample_id values")
    require(not frame.relative_path.duplicated().any(), "Duplicate relative_path values")
    require(frame.usage.isin(TRAIN_USAGES | NON_TRAIN_USAGES).all(), "Unknown usage values")
    require(
        (frame[frame.usage == "fit"].official_partition == "train").all(),
        "Non-train partition entered fit",
    )
    require(
        frame.loc[frame.usage == "validation", "official_partition"]
        .isin({"train", "validation"})
        .all(),
        "Validation must come from train or an official validation partition",
    )
    require((frame[frame.usage == "test"].official_partition == "test").all(), "Test usage mismatch")
    require((frame[frame.usage == "evaluation"].official_partition == "evaluation").all(), "Evaluation usage mismatch")
    require((frame[frame.usage.isin(TRAIN_USAGES)]["class_index"] >= 0).all(), "Unlabeled training sample")
    source = frame[frame.stage == 0]
    if not source.empty:
        require(source.domain.nunique() == 1, "Stage 0 must contain one source domain")
        require((source.domain_index == 0).all(), "Stage-0 source must have domain_index 0")
    stream = frame[frame.stage > 0]
    require((stream.stage >= 1).all(), "Incremental stages must be positive")
    require(
        (stream.groupby("domain").stage.nunique() == 1).all(),
        "A domain is assigned to multiple stream stages",
    )
    require(
        (stream.groupby("stage").domain.nunique() == 1).all(),
        "A stream stage contains multiple domains",
    )

    leakage = {}
    for first, second in (("fit", "validation"), ("fit", "test"), ("validation", "test")):
        recording_overlap = intersections(frame, "recording_group", first, second)
        content_overlap = intersections(frame, "content_group", first, second)
        require(not recording_overlap, f"Recording-group leakage {first}<->{second}: {len(recording_overlap)}")
        require(not content_overlap, f"Content-group leakage {first}<->{second}: {len(content_overlap)}")
        leakage[f"{first}_vs_{second}"] = {
            "recording_overlap": len(recording_overlap),
            "content_overlap": len(content_overlap),
        }

    cross_domain_content = (
        frame[frame.usage.isin(TRAIN_USAGES)]
        .groupby("content_group")
        .domain.nunique()
    )
    cross_domain_count = int((cross_domain_content > 1).sum())
    if args.forbid_cross_domain_content_overlap:
        require(cross_domain_count == 0, f"Found {cross_domain_count} content groups shared across domains")

    duplicate_content_groups = {}
    for usage, usage_frame in frame.groupby("usage"):
        sizes = usage_frame.groupby("content_group").size()
        duplicate_content_groups[str(usage)] = {
            "groups": int((sizes > 1).sum()),
            "extra_rows": int((sizes[sizes > 1] - 1).sum()),
        }

    if args.require_files:
        if args.data_root is None:
            raise ValueError("--data-root is required with --require-files")
        missing = [str(path) for path in absolute_paths(frame, args.data_root) if not path.is_file()]
        require(not missing, f"Missing {len(missing)} audio files; first={missing[:3]}")

    class_coverage = (
        frame[frame.usage.isin(TRAIN_USAGES)]
        .groupby(["domain", "usage"])
        .label.nunique()
    )
    domains = sorted(frame.loc[frame.domain_index >= 0, "domain"].unique())
    report = {
        "status": "passed",
        "manifest": str(args.manifest),
        "required_columns": REQUIRED_COLUMNS,
        "rows": len(frame),
        "domains": domains,
        "usage_counts": {
            str(key): int(value) for key, value in frame.usage.value_counts().sort_index().items()
        },
        "class_coverage": {
            f"{domain}/{usage}": int(value)
            for (domain, usage), value in class_coverage.items()
        },
        "split_leakage": leakage,
        "cross_domain_content_groups": cross_domain_count,
        "within_usage_duplicate_content": duplicate_content_groups,
        "has_pcm_fingerprints": "pcm_sha256" in frame.columns,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
