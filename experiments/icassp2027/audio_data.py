from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 32_000
CLIP_SAMPLES = SAMPLE_RATE * 4
DATASET_CLIP_SAMPLES = {
    "DIL-DCASE26": SAMPLE_RATE * 4,
    "TAU-ASC2022-Mobile": SAMPLE_RATE,
    "ADIL-Europe-Korea-Fixed4": SAMPLE_RATE * 10,
}


class ManifestWaveformDataset(Dataset):
    """Audio dataset whose complete file list is supplied by the protocol guard."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        clip_samples: int | None = None,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = data_root.resolve()
        datasets = set(self.frame.dataset.astype(str))
        if clip_samples is None:
            if len(datasets) != 1:
                raise ValueError(f"One loader cannot mix dataset clip policies: {sorted(datasets)}")
            clip_samples = DATASET_CLIP_SAMPLES.get(next(iter(datasets)), CLIP_SAMPLES)
        if clip_samples <= 0:
            raise ValueError("clip_samples must be positive")
        self.clip_samples = int(clip_samples)
        self.paths = [(self.data_root / value).resolve() for value in self.frame.relative_path]
        outside = [path for path in self.paths if not path.is_relative_to(self.data_root)]
        if outside:
            raise PermissionError(f"Manifest paths escape data root: {outside[:3]}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        path = self.paths[index]
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(
                np.asarray(waveform), orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if len(waveform) < self.clip_samples:
            waveform = np.pad(waveform, (0, self.clip_samples - len(waveform)))
        else:
            waveform = waveform[: self.clip_samples]
        return torch.from_numpy(waveform), int(row.class_index), str(row.sample_id)
