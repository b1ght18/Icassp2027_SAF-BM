from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .manifest import REQUIRED_COLUMNS, stable_id, write_manifest


META_COLUMNS = ["relative_path", "label", "domain", "class_index"]


def rows_from_metadata(path: Path, partition: str) -> list[dict[str, object]]:
    metadata = pd.read_csv(path, sep="\t", names=META_COLUMNS)
    rows = []
    for item in metadata.itertuples(index=False):
        domain_index = int(str(item.domain).removeprefix("D")) - 1
        rows.append(
            {
                "sample_id": stable_id("dil-dcase26", item.relative_path),
                "relative_path": str(item.relative_path),
                "label": str(item.label),
                "class_index": int(item.class_index),
                "domain": str(item.domain),
                "domain_index": domain_index,
                "stage": domain_index,
                "official_partition": partition,
                "usage": "unassigned" if partition == "train" else "test",
                "recording_group": stable_id("recording", item.relative_path),
                "content_group": stable_id("content", item.relative_path),
                "dataset": "DIL-DCASE26",
                "source_domain_available": False,
            }
        )
    return rows


def evaluation_rows(audio_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(audio_root.glob("*.wav")):
        relative_path = str(path.relative_to(audio_root.parent))
        rows.append(
            {
                "sample_id": stable_id("dil-dcase26-eval", relative_path),
                "relative_path": relative_path,
                "label": pd.NA,
                "class_index": -1,
                "domain": "UNKNOWN",
                "domain_index": -1,
                "stage": -1,
                "official_partition": "evaluation",
                "usage": "evaluation",
                "recording_group": stable_id("recording", relative_path),
                "content_group": stable_id("content", relative_path),
                "dataset": "DIL-DCASE26",
                "source_domain_available": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("task7_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027/manifests/dcase_base.tsv"),
    )
    parser.add_argument("--include-evaluation", action="store_true")
    args = parser.parse_args()
    setup = args.data_root / "evaluation_setup"
    rows = [
        *rows_from_metadata(setup / "development_train.txt", "train"),
        *rows_from_metadata(setup / "development_test.txt", "test"),
    ]
    if args.include_evaluation:
        rows.extend(evaluation_rows(args.data_root / "dil-dcase26-eval"))
    frame = pd.DataFrame(rows)
    write_manifest(frame, args.output)
    print(frame.groupby(["official_partition", "domain"], dropna=False).size().to_string())
    print(f"Saved {len(frame)} rows and {len(REQUIRED_COLUMNS)} required fields to {args.output}")


if __name__ == "__main__":
    main()
