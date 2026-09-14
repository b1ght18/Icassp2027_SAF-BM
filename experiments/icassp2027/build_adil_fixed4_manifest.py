"""Build leakage-audited fixed-label Europe-to-Korea manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .make_incremental_splits import assign_grouped_validation
    from .manifest import stable_id, write_manifest
except ImportError:
    from make_incremental_splits import assign_grouped_validation
    from manifest import stable_id, write_manifest


TARGET_LABELS = ("bus", "metro", "metro_station", "park")
LABEL_TO_INDEX = {label: index for index, label in enumerate(TARGET_LABELS)}
CITY_TO_DOMAIN = {
    "barcelona": "D1",
    "helsinki": "D1",
    "london": "D1",
    "paris": "D1",
    "stockholm": "D1",
    "vienna": "D1",
    "lisbon": "D2",
    "lyon": "D3",
    "prague": "D4",
}
KOREA_CLASS_MAP = {
    "Bus": "bus",
    "Park": "park",
    "Subway": "metro",
    "SubwayStation": "metro_station",
}


def read_fold(path: Path, partition: str, header: bool) -> pd.DataFrame:
    if header:
        frame = pd.read_csv(path, sep="\t")
    else:
        frame = pd.read_csv(path, sep="\t", names=["filename", "scene_label"])
    frame["official_partition"] = partition
    return frame


def parse_europe_name(filename: str) -> dict[str, str]:
    parts = Path(filename).stem.split("-")
    if len(parts) != 5:
        raise ValueError(f"unexpected European filename: {filename}")
    scene, city, location, recording, device = parts
    if device != "a":
        raise ValueError(f"fixed-four protocol expects device a: {filename}")
    return {
        "scene": scene,
        "city": city,
        "location": location,
        "recording": recording,
        "device": device,
    }


def europe_rows(
    dataset: str,
    frames: list[pd.DataFrame],
    relative_root: Path,
) -> list[dict[str, object]]:
    rows = []
    frame = pd.concat(frames, ignore_index=True)
    for item in frame.itertuples(index=False):
        label = str(item.scene_label)
        if label not in LABEL_TO_INDEX:
            continue
        parsed = parse_europe_name(str(item.filename))
        if parsed["city"] not in CITY_TO_DOMAIN:
            continue
        if dataset == "TUT2018" and CITY_TO_DOMAIN[parsed["city"]] != "D1":
            continue
        if dataset == "TAU2019" and CITY_TO_DOMAIN[parsed["city"]] == "D1":
            continue
        domain = CITY_TO_DOMAIN[parsed["city"]]
        domain_index = int(domain[1:]) - 1
        basename = Path(str(item.filename)).name
        relative_path = (relative_root / dataset / basename).as_posix()
        location_group = (
            f"{dataset}:{label}:{parsed['city']}:{parsed['location']}"
        )
        recording_group = f"{location_group}:{parsed['recording']}"
        partition = str(item.official_partition)
        rows.append(
            {
                "sample_id": stable_id(dataset, basename),
                "relative_path": relative_path,
                "label": label,
                "class_index": LABEL_TO_INDEX[label],
                "domain": domain,
                "domain_index": domain_index,
                "stage": domain_index,
                "official_partition": partition,
                "usage": "unassigned" if partition == "train" else "test",
                "recording_group": stable_id("location", location_group),
                "content_group": stable_id("recording", recording_group),
                "dataset": "ADIL-Europe-Korea-Fixed4",
                "source_corpus": dataset,
                "city": parsed["city"],
                "location": parsed["location"],
                "recording": parsed["recording"],
                "source_domain_available": True,
            }
        )
    return rows


def korea_rows(korea_root: Path, relative_root: Path) -> list[dict[str, object]]:
    rows = []
    for split, partition, usage in (
        ("Train", "train", "fit"),
        ("Val", "validation", "validation"),
        ("Test", "test", "test"),
    ):
        for archive_class, label in KOREA_CLASS_MAP.items():
            class_root = korea_root / split / archive_class
            for path in sorted(class_root.glob("*.wav")):
                parts = path.stem.split("_")
                if len(parts) < 4:
                    raise ValueError(f"unexpected CochlScene filename: {path.name}")
                user = parts[1]
                recording = parts[2]
                segment = "_".join(parts[3:])
                group = f"CochlScene:{label}:{user}:{recording}"
                relative_path = (
                    relative_root / split / archive_class / path.name
                ).as_posix()
                rows.append(
                    {
                        "sample_id": stable_id("CochlScene", split, path.name),
                        "relative_path": relative_path,
                        "label": label,
                        "class_index": LABEL_TO_INDEX[label],
                        "domain": "D5",
                        "domain_index": 4,
                        "stage": 4,
                        "official_partition": partition,
                        "usage": usage,
                        "recording_group": stable_id("cochl-recording", group),
                        "content_group": stable_id(
                            "cochl-segment", group, segment
                        ),
                        "dataset": "ADIL-Europe-Korea-Fixed4",
                        "source_corpus": "CochlScene",
                        "city": "korea",
                        "location": user,
                        "recording": recording,
                        "source_domain_available": True,
                    }
                )
    return rows


def split_europe(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    for domain_index, domain_frame in frame[
        (frame.domain_index < 4) & (frame.official_partition == "train")
    ].groupby("domain_index", sort=True):
        assignments = assign_grouped_validation(
            domain_frame,
            fraction,
            seed + 1009 * int(domain_index),
        )
        frame.loc[domain_frame.index, "usage"] = assignments
    return frame


def audit(frame: pd.DataFrame, data_root: Path, check_files: bool) -> dict[str, object]:
    overlaps = {}
    usages = ("fit", "validation", "test")
    for index, first in enumerate(usages):
        for second in usages[index + 1 :]:
            a = set(frame.loc[frame.usage == first, "recording_group"])
            b = set(frame.loc[frame.usage == second, "recording_group"])
            overlaps[f"{first}_{second}"] = len(a & b)
    missing = []
    if check_files:
        missing = [
            value
            for value in frame.relative_path
            if not (data_root / value).is_file()
        ]
    return {
        "rows": len(frame),
        "counts": {
            f"{domain}:{usage}": int(count)
            for (domain, usage), count in frame.groupby(["domain", "usage"]).size().items()
        },
        "class_counts": {
            f"{domain}:{usage}:{label}": int(count)
            for (domain, usage, label), count in frame.groupby(
                ["domain", "usage", "label"]
            ).size().items()
        },
        "recording_group_overlaps": overlaps,
        "missing_files": len(missing),
        "missing_examples": missing[:10],
        "status": "passed"
        if not missing and all(value == 0 for value in overlaps.values())
        else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tut-train", type=Path, required=True)
    parser.add_argument("--tut-evaluate", type=Path, required=True)
    parser.add_argument("--tau-train", type=Path, required=True)
    parser.add_argument("--tau-evaluate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--europe-relative-root", type=Path, required=True)
    parser.add_argument("--korea-root", type=Path, required=True)
    parser.add_argument("--korea-relative-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    rows = europe_rows(
        "TUT2018",
        [
            read_fold(args.tut_train, "train", False),
            read_fold(args.tut_evaluate, "test", False),
        ],
        args.europe_relative_root,
    )
    rows.extend(
        europe_rows(
            "TAU2019",
            [
                read_fold(args.tau_train, "train", True),
                read_fold(args.tau_evaluate, "test", True),
            ],
            args.europe_relative_root,
        )
    )
    rows.extend(korea_rows(args.korea_root, args.korea_relative_root))
    frame = pd.DataFrame(rows)
    expected_corpus_counts = {
        "TUT2018": 3456,
        "TAU2019": 1728,
        "CochlScene": 23359,
    }
    observed_corpus_counts = frame.groupby("source_corpus").size().to_dict()
    if observed_corpus_counts != expected_corpus_counts:
        raise RuntimeError(
            f"Incomplete fixed-four corpus: {observed_corpus_counts} != "
            f"{expected_corpus_counts}"
        )
    frame = split_europe(frame, args.validation_fraction, args.seed)
    frame["split_seed"] = args.seed
    frame["validation_fraction"] = args.validation_fraction
    write_manifest(frame, args.output)
    report = audit(frame, args.data_root, args.check_files)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("manifest audit failed")


if __name__ == "__main__":
    main()
