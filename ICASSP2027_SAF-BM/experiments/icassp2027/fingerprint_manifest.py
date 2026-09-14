from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from .manifest import load_manifest, stable_id, write_manifest
except ImportError:
    from manifest import load_manifest, stable_id, write_manifest


def digest_file(path: Path) -> tuple[str, str, int, int]:
    byte_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            byte_digest.update(block)
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # Quantizing decoded PCM makes the fingerprint independent of WAV metadata while
    # remaining exact at 16-bit audio precision. Channel count and sample rate are salted.
    quantized = np.clip(np.rint(waveform * 32767.0), -32768, 32767).astype("<i2")
    pcm_digest = hashlib.sha256()
    pcm_digest.update(f"sr={sample_rate};channels={quantized.shape[1]};".encode())
    pcm_digest.update(quantized.tobytes(order="C"))
    return byte_digest.hexdigest(), pcm_digest.hexdigest(), int(sample_rate), len(quantized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add content fingerprints to an audio manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    frame = load_manifest(args.manifest).copy()
    paths = [(args.data_root / relative).resolve() for relative in frame.relative_path]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files; first={missing[:3]}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fingerprints = list(pool.map(digest_file, paths, chunksize=16))
    frame["byte_sha256"] = [value[0] for value in fingerprints]
    frame["pcm_sha256"] = [value[1] for value in fingerprints]
    frame["sample_rate"] = [value[2] for value in fingerprints]
    frame["audio_frames"] = [value[3] for value in fingerprints]
    frame["content_group"] = [stable_id("pcm", value[1]) for value in fingerprints]
    write_manifest(frame, args.output)
    duplicate_pcm = frame.groupby("pcm_sha256").size()
    print(
        f"rows={len(frame)} unique_pcm={frame.pcm_sha256.nunique()} "
        f"duplicate_pcm_groups={int((duplicate_pcm > 1).sum())}"
    )


if __name__ == "__main__":
    main()
