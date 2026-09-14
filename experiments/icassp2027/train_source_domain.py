from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .audio_data import ManifestWaveformDataset
    from .dcase_boundary_model import make_backbone
    from .manifest import load_manifest
    from .train_dcase_stage import choose_device, macro_accuracy, set_seed
except ImportError:
    from audio_data import ManifestWaveformDataset
    from dcase_boundary_model import make_backbone
    from manifest import load_manifest
    from train_dcase_stage import choose_device, macro_accuracy, set_seed


def run_epoch(model, loader, device, optimizer=None, max_batches: int | None = None):
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (waveforms, labels, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms, task=0)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "macro_accuracy": macro_accuracy(targets, predictions),
        "samples": len(targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the D1 source anchor for a second leakage-safe dataset"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    frame = load_manifest(args.manifest)
    source = frame[frame.domain_index == 0]
    fit = source[source.usage == "fit"]
    validation = source[source.usage == "validation"]
    if fit.empty or validation.empty:
        raise RuntimeError("D1 source requires non-empty fit and validation rows")
    class_indices = sorted(frame.class_index.unique().tolist())
    if class_indices != list(range(len(class_indices))):
        raise RuntimeError(f"Class indices must be contiguous: {class_indices}")
    classes_num = len(class_indices)
    nb_tasks = int(frame.domain_index.max()) + 1
    device = choose_device(args.device)
    model = make_backbone(classes_num=classes_num, nb_tasks=nb_tasks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    def make_loader(rows, shuffle):
        generator = torch.Generator().manual_seed(args.seed) if shuffle else None
        return DataLoader(
            ManifestWaveformDataset(rows, args.data_root),
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )

    fit_loader = make_loader(fit, True)
    validation_loader = make_loader(validation, False)
    best_state = None
    best_accuracy, best_loss, best_epoch, stale = -math.inf, math.inf, 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, fit_loader, device, optimizer, args.max_train_batches
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, None, args.max_validation_batches
        )
        scheduler.step()
        if (
            validation_metrics["macro_accuracy"],
            -validation_metrics["loss"],
        ) > (best_accuracy, -best_loss):
            best_state = copy.deepcopy(model.state_dict())
            best_accuracy = validation_metrics["macro_accuracy"]
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "elapsed_seconds": time.time() - start,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Source training produced no checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "icassp2027-source-anchor-v1",
            "model_state_dict": best_state,
            "metadata": {
                "dataset": sorted(source.dataset.unique().tolist()),
                "manifest": str(args.manifest.resolve()),
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_validation_macro_accuracy": best_accuracy,
                "source_domain": sorted(source.domain.unique().tolist()),
                "fit_samples": len(fit),
                "validation_samples": len(validation),
                "classes_num": classes_num,
                "nb_tasks": nb_tasks,
            },
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "best_epoch": best_epoch,
                "best_validation_macro_accuracy": best_accuracy,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
