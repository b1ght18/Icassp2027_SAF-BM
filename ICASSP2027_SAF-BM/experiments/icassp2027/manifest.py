from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "sample_id",
    "relative_path",
    "label",
    "class_index",
    "domain",
    "domain_index",
    "stage",
    "official_partition",
    "usage",
    "recording_group",
    "content_group",
    "dataset",
]

TRAIN_USAGES = {"fit", "validation"}
NON_TRAIN_USAGES = {"test", "evaluation", "unassigned"}


def stable_id(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"label": "string"})
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path} misses columns: {sorted(missing)}")
    return frame


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot write manifest without: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = REQUIRED_COLUMNS + [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    frame[ordered].to_csv(path, sep="\t", index=False)


def absolute_paths(frame: pd.DataFrame, data_root: Path) -> list[Path]:
    return [(data_root / value).resolve() for value in frame.relative_path]
