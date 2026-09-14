from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from .manifest import stable_id, write_manifest
except ImportError:
    from manifest import stable_id, write_manifest


TAU_PATTERN = re.compile(
    # TAU 2022 one-second clips use
    # scene-city-location-recording-segment-device.wav.  Keeping both the
    # recording and one-second segment is necessary to tie simultaneous
    # device captures without accidentally merging unrelated recordings.
    r"^(?P<scene>.+)-(?P<city>[^-]+)-(?P<location>[^-]+)-"
    r"(?P<recording>[^-]+)-(?P<segment>[^-]+)-(?P<device>[^-.]+)\.wav$"
)


def parse_audio_name(relative_path: str) -> dict[str, str]:
    match = TAU_PATTERN.match(Path(relative_path).name)
    if match is None:
        raise ValueError(f"Cannot parse TAU filename: {relative_path}")
    return match.groupdict()


def read_split(path: Path, partition: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "filename" not in frame.columns:
        frame = pd.read_csv(path, sep="\t", names=["filename", "scene_label"])
    if "scene_label" not in frame.columns:
        raise ValueError(f"Labeled split required, missing scene_label in {path}")
    frame["official_partition"] = partition
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domain-map",
        default="a:D1,b:D2,c:D3",
        help="Comma-separated device:domain mapping.",
    )
    parser.add_argument(
        "--strip-path-prefix",
        default="audio/",
        help="Prefix present in the official metadata but absent below data-root.",
    )
    args = parser.parse_args()
    device_to_domain = {
        device.strip().lower(): domain.strip().upper()
        for device, domain in (item.split(":", 1) for item in args.domain_map.split(","))
    }
    frames = [read_split(args.train_split, "train"), read_split(args.test_split, "test")]
    labels = sorted(set(pd.concat(frames).scene_label.astype(str)))
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows = []
    for frame in frames:
        for item in frame.itertuples(index=False):
            metadata_path = str(item.filename).replace("\\", "/")
            parsed = parse_audio_name(metadata_path)
            device = parsed["device"].lower()
            if device not in device_to_domain:
                continue
            relative_path = metadata_path
            if args.strip_path_prefix and relative_path.startswith(args.strip_path_prefix):
                relative_path = relative_path[len(args.strip_path_prefix) :]
            domain = device_to_domain[device]
            domain_index = int(domain.removeprefix("D")) - 1
            location_group = f"{parsed['scene']}:{parsed['city']}:{parsed['location']}"
            original_recording_group = f"{location_group}:{parsed['recording']}"
            simultaneous_group = f"{original_recording_group}:{parsed['segment']}"
            rows.append(
                {
                    "sample_id": stable_id("tau2022", relative_path),
                    "relative_path": relative_path,
                    "label": str(item.scene_label),
                    "class_index": label_to_index[str(item.scene_label)],
                    "domain": domain,
                    "domain_index": domain_index,
                    "stage": domain_index,
                    "official_partition": str(item.official_partition),
                    "usage": "unassigned" if item.official_partition == "train" else "test",
                    "recording_group": stable_id("location", location_group),
                    "content_group": stable_id("simultaneous", simultaneous_group),
                    "dataset": "TAU-ASC2022-Mobile",
                    "device": device,
                    "city": parsed["city"],
                    "location": parsed["location"],
                    "recording": parsed["recording"],
                    "segment": parsed["segment"],
                    "original_recording_group": stable_id(
                        "original-recording", original_recording_group
                    ),
                    "source_domain_available": True,
                }
            )
    output = pd.DataFrame(rows)
    write_manifest(output, args.output)
    print(output.groupby(["official_partition", "domain"]).size().to_string())
    print(f"Saved {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
