from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .audio_data import ManifestWaveformDataset
    from .dcase_boundary_model import BoundaryMigrationModel, load_source_backbone
    from .protocol_guard import StageAccessGuard
    from .train_dcase_stage import choose_device
except ImportError:
    from audio_data import ManifestWaveformDataset
    from dcase_boundary_model import BoundaryMigrationModel, load_source_backbone
    from protocol_guard import StageAccessGuard
    from train_dcase_stage import choose_device


def normalization_overlay(path: Path, task: int) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("incremental_state_dict", raw.get("model_state_dict", raw))
    selected = {}
    for name, value in state.items():
        normalized = name if name.startswith("backbone.") else f"backbone.{name}"
        if any(token in normalized for token in (f"bn0.{task}.", f"bnF.{task}.", f"bnS.{task}.")):
            selected[normalized] = value
    if not selected:
        raise RuntimeError(f"No task-{task} normalization state in {path}")
    return selected


def extract(
    model: BoundaryMigrationModel,
    loader: DataLoader,
    task: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, sample_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for waveforms, batch_labels, batch_ids in loader:
            batch_features = model.features(waveforms.to(device, non_blocking=True), task)
            features.append(batch_features.cpu().numpy().astype(np.float32, copy=False))
            labels.append(batch_labels.numpy())
            sample_ids.extend(batch_ids)
    return np.concatenate(features), np.concatenate(labels).astype(np.int64), np.asarray(sample_ids)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Cache current-domain frozen normalized features")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--normalization-checkpoint", type=Path)
    parser.add_argument(
        "--feature-normalization", choices=("target", "source"), default="target"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=root / "task7_data")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=root / "model/baseline/checkpoints/checkpoint_D1.pth",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    audit = args.output.with_suffix(".access.json")
    with StageAccessGuard(args.manifest, args.stage, audit) as guard:
        fit_frame = guard.current("fit")
        validation_frame = guard.current("validation")
        task_values = sorted(fit_frame.domain_index.unique().tolist())
        if len(task_values) != 1:
            raise RuntimeError("Stage must contain one domain")
        task = int(task_values[0])
        backbone = load_source_backbone(args.source_checkpoint)
        model = BoundaryMigrationModel(backbone, method="fixed", rank=1)
        if args.feature_normalization == "target":
            if args.normalization_checkpoint is None:
                parser.error("target feature normalization requires --normalization-checkpoint")
            model.load_state_dict(
                normalization_overlay(args.normalization_checkpoint, task), strict=False
            )
            feature_task = task
        else:
            feature_task = 0
        device = choose_device(args.device)
        model.to(device).eval()

        def loader(frame):
            return DataLoader(
                ManifestWaveformDataset(frame, args.data_root),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                persistent_workers=args.workers > 0,
                pin_memory=device.type == "cuda",
            )

        fit = extract(model, loader(fit_frame), feature_task, device)
        validation = extract(model, loader(validation_frame), feature_task, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            fit_features=fit[0],
            fit_labels=fit[1],
            fit_sample_ids=fit[2],
            validation_features=validation[0],
            validation_labels=validation[1],
            validation_sample_ids=validation[2],
            source_weight=model.backbone.fc.weight.detach().cpu().numpy(),
            source_bias=model.backbone.fc.bias.detach().cpu().numpy(),
            task=np.asarray(task),
            stage=np.asarray(args.stage),
            feature_task=np.asarray(feature_task),
            feature_normalization=np.asarray(args.feature_normalization),
            manifest=np.asarray(str(args.manifest.resolve())),
            normalization_checkpoint=np.asarray(
                str(args.normalization_checkpoint.resolve())
                if args.normalization_checkpoint is not None
                else ""
            ),
        )
        print(
            {"output": str(args.output), "task": task, "feature_task": feature_task,
             "fit": len(fit[1]), "validation": len(validation[1]), "device": str(device)}
        )


if __name__ == "__main__":
    main()
