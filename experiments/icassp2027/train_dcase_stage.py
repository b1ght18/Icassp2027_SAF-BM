from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .audio_data import ManifestWaveformDataset
    from .boundary_residual import (
        AdaptiveBoundaryResidual,
        OrderedAdaptiveBoundaryResidual,
        SpectralOrderedBoundaryResidual,
    )
    from .dcase_boundary_model import (
        BoundaryMigrationModel,
        copy_normalization_branch,
        incremental_state_dict,
        load_incremental_state,
        load_source_backbone,
        tensor_subset_hash,
    )
    from .protocol_guard import StageAccessGuard
except ImportError:
    from audio_data import ManifestWaveformDataset
    from boundary_residual import (
        AdaptiveBoundaryResidual,
        OrderedAdaptiveBoundaryResidual,
        SpectralOrderedBoundaryResidual,
    )
    from dcase_boundary_model import (
        BoundaryMigrationModel,
        copy_normalization_branch,
        incremental_state_dict,
        load_incremental_state,
        load_source_backbone,
        tensor_subset_hash,
    )
    from protocol_guard import StageAccessGuard


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def macro_accuracy(targets: list[int], predictions: list[int]) -> float:
    scores = []
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    for class_index in sorted(set(targets)):
        mask = target_array == class_index
        scores.append(float(np.mean(prediction_array[mask] == class_index)))
    return 100.0 * float(np.mean(scores)) if scores else 0.0


def run_epoch(
    model: BoundaryMigrationModel,
    loader: DataLoader,
    task: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    l0_lambda: float,
    orthogonality_lambda: float,
    max_batches: int | None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.set_stage_mode(task, training)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    head = model.heads[str(task)]
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (waveforms, labels, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms, task)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            if isinstance(
                head,
                (AdaptiveBoundaryResidual, OrderedAdaptiveBoundaryResidual, SpectralOrderedBoundaryResidual),
            ):
                regularization = head.regularization()
                loss = loss + l0_lambda * (
                    regularization.expected_active_components / head.max_rank
                )
                loss = loss + orthogonality_lambda * regularization.orthogonality
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "macro_accuracy": macro_accuracy(targets, predictions),
        "samples": len(targets),
    }


def save_checkpoint(
    path: Path,
    model: BoundaryMigrationModel,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "icassp2027-boundary-migration-v2-incremental",
            "incremental_state_dict": incremental_state_dict(
                model, tuple(int(task) for task in metadata["seen_tasks"])
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def collect_rp32_class_prototypes(
    model: BoundaryMigrationModel,
    loader: DataLoader,
    device: torch.device,
    projection_seed: int = 8_675_309,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor | int]:
    """Store only class-conditional routing sufficient statistics, never examples."""
    generator = torch.Generator().manual_seed(projection_seed)
    projection = torch.randn(2048, 32, generator=generator) / np.sqrt(32.0)
    projection = projection.to(device)
    # Accumulate on CPU. MPS index_add_ can report corrupted asynchronous indices
    # even when labels are valid; only feature extraction/projection needs acceleration.
    classes_num = model.backbone.fc.out_features
    sums = torch.zeros(classes_num, 32)
    counts = torch.zeros(classes_num, dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for batch_index, (waveforms, labels, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            waveforms = waveforms.to(device, non_blocking=True)
            features = model.features(waveforms, task=0)
            projected = F.normalize(features @ projection, dim=1).cpu()
            labels = labels.cpu()
            sums.index_add_(0, labels, projected)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
    valid = counts > 0
    # Boolean advanced indexing is unreliable on some MPS versions. Normalizing
    # every row is equivalent because F.normalize maps an all-zero row to zero.
    prototypes = F.normalize(sums, dim=1) * valid.unsqueeze(1).to(sums.dtype)
    return {
        "class_prototypes": prototypes,
        "class_valid": valid,
        "counts": counts,
        "projection_seed": projection_seed,
        "projection_dimension": 32,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Leakage-guarded sequential training of W_t = W_0 + Delta W_t"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=root / "task7_data")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=root / "model/baseline/checkpoints/checkpoint_D1.pth",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--normalization-checkpoint",
        type=Path,
        help="Overlay only the current task BN state before residual-only training.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--last-output",
        type=Path,
        help="Also retain the final epoch while --output retains validation-best.",
    )
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=("bn_only", "fixed", "adaptive", "adaptive_ordered", "spectral_ordered"),
        default="adaptive_ordered"
    )
    parser.add_argument("--freeze-normalization", action="store_true")
    parser.add_argument(
        "--freeze-boundary",
        action="store_true",
        help="Train only the current normalization branch with a fixed initialized head.",
    )
    parser.add_argument(
        "--head-init-checkpoint",
        type=Path,
        help="Validation-only cached head candidates used to initialize the current task head.",
    )
    parser.add_argument("--normalization-init", choices=("source", "checkpoint"), default="source")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--initial-log-alpha", type=float, default=2.0)
    parser.add_argument("--initial-rank", type=float)
    parser.add_argument("--l0-lambda", type=float, default=0.01)
    parser.add_argument("--orthogonality-lambda", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--boundary-learning-rate",
        type=float,
        help=(
            "Optional learning rate for the current boundary head. When set, the "
            "normalization parameters retain --learning-rate and the optimizer uses "
            "a separate boundary parameter group."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=("adamw", "adam", "official_adam"), default="adamw")
    parser.add_argument("--class-balanced-loss", action="store_true")
    parser.add_argument(
        "--checkpoint-selection", choices=("validation", "last"), default="validation"
    )
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-router-batches", type=int)
    args = parser.parse_args()

    if args.stage < 1:
        parser.error("Incremental stage must be at least 1")
    if args.stage > 1 and args.resume is None:
        parser.error("Stages after 1 require --resume from the preceding checkpoint")
    if args.last_output is not None and args.checkpoint_selection != "validation":
        parser.error("--last-output is only meaningful with validation checkpoint selection")
    if args.method == "bn_only" and args.freeze_normalization:
        parser.error("bn_only with --freeze-normalization has no trainable parameters")
    if args.freeze_boundary and args.method == "bn_only":
        parser.error("--freeze-boundary requires a boundary-capable method")
    if args.freeze_boundary and args.head_init_checkpoint is None:
        parser.error("--freeze-boundary requires --head-init-checkpoint")
    set_seed(args.seed)
    device = choose_device(args.device)
    audit_path = args.audit_output or args.output.with_suffix(".access.json")
    with StageAccessGuard(args.manifest, args.stage, audit_path) as guard:
        fit_frame = guard.current("fit")
        validation_frame = guard.current("validation")
        current_tasks = sorted(fit_frame.domain_index.unique().tolist())
        if len(current_tasks) != 1 or current_tasks != sorted(
            validation_frame.domain_index.unique().tolist()
        ):
            raise RuntimeError("Each incremental stage must contain exactly one consistent domain")
        task = int(current_tasks[0])
        guard.reject_paths(fit_frame.relative_path.astype(str).tolist())
        guard.reject_paths(validation_frame.relative_path.astype(str).tolist())

        backbone = load_source_backbone(args.source_checkpoint)
        if args.stage >= len(backbone.bn0):
            raise RuntimeError(
                f"Stage {args.stage} exceeds checkpoint task count {len(backbone.bn0)}"
            )
        if args.normalization_init == "source":
            for target_task in range(1, len(backbone.bn0)):
                copy_normalization_branch(backbone, 0, target_task)
        model = BoundaryMigrationModel(
            backbone,
            method=args.method,
            rank=args.rank,
            max_rank=args.max_rank,
            alpha=args.alpha,
            initial_log_alpha=args.initial_log_alpha,
            initial_rank=args.initial_rank,
        )
        previous_router_state = {}
        if args.resume is not None:
            resume = torch.load(args.resume, map_location="cpu", weights_only=False)
            resume_metadata = resume.get("metadata", {})
            if resume_metadata.get("stage") != args.stage - 1:
                raise RuntimeError(
                    f"Resume checkpoint stage={resume_metadata.get('stage')}; expected {args.stage - 1}"
                )
            if resume_metadata.get("method") != args.method:
                raise RuntimeError("Resume method does not match requested method")
            if bool(resume_metadata.get("freeze_normalization")) != args.freeze_normalization:
                raise RuntimeError("Resume normalization ablation does not match requested run")
            if bool(resume_metadata.get("freeze_boundary", False)) != args.freeze_boundary:
                raise RuntimeError("Resume boundary-freezing ablation does not match requested run")
            if resume_metadata.get("normalization_init") != args.normalization_init:
                raise RuntimeError("Resume normalization initialization does not match")
            state = resume.get("incremental_state_dict", resume.get("model_state_dict"))
            if state is None:
                raise RuntimeError("Resume checkpoint has no model state")
            load_incremental_state(model, state)
            previous_router_state = resume.get("router_state", {})

        if args.head_init_checkpoint is not None:
            head_payload = torch.load(
                args.head_init_checkpoint, map_location="cpu", weights_only=False
            )
            if int(head_payload.get("stage", -1)) != args.stage:
                raise RuntimeError("Head initializer stage does not match the current stage")
            if int(head_payload.get("task", -1)) != task:
                raise RuntimeError("Head initializer task does not match the current domain")
            states = head_payload.get("all_states", {})
            head_state = states.get(args.rank, states.get(str(args.rank)))
            if head_state is None:
                raise RuntimeError(
                    f"Rank {args.rank} is unavailable in {args.head_init_checkpoint}"
                )
            model.heads[str(task)].load_state_dict(head_state, strict=True)

        if args.normalization_checkpoint is not None:
            normalization_raw = torch.load(
                args.normalization_checkpoint, map_location="cpu", weights_only=False
            )
            normalization_state = normalization_raw.get(
                "incremental_state_dict",
                normalization_raw.get("model_state_dict", normalization_raw),
            )
            selected_normalization = {}
            for name, value in normalization_state.items():
                normalized_name = name if name.startswith("backbone.") else f"backbone.{name}"
                if any(
                    token in normalized_name
                    for token in (f"bn0.{task}.", f"bnF.{task}.", f"bnS.{task}.")
                ):
                    selected_normalization[normalized_name] = value
            if not selected_normalization:
                raise RuntimeError(
                    f"No task-{task} normalization state in {args.normalization_checkpoint}"
                )
            model.load_state_dict(selected_normalization, strict=False)

        model.activate_stage(
            task,
            freeze_normalization=args.freeze_normalization,
            freeze_boundary=args.freeze_boundary,
        )
        model.to(device)
        previously_seen = tuple(resume_metadata.get("seen_tasks", [])) if args.resume else ()
        old_tasks = (0, *previously_seen)
        old_hash_before = tensor_subset_hash(model, old_tasks)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_parameters: list[torch.Tensor] | list[dict[str, object]] = parameters
        if args.boundary_learning_rate is not None:
            if args.method == "bn_only" or args.freeze_boundary:
                parser.error("--boundary-learning-rate requires a trainable boundary head")
            boundary_parameters = [
                parameter
                for parameter in model.heads[str(task)].parameters()
                if parameter.requires_grad
            ]
            boundary_ids = {id(parameter) for parameter in boundary_parameters}
            normalization_parameters = [
                parameter for parameter in parameters if id(parameter) not in boundary_ids
            ]
            if not boundary_parameters or not normalization_parameters:
                raise RuntimeError(
                    "Separate boundary learning rate requires both trainable normalization "
                    "and boundary parameters"
                )
            optimizer_parameters = [
                {"params": normalization_parameters, "lr": args.learning_rate},
                {"params": boundary_parameters, "lr": args.boundary_learning_rate},
            ]
        if args.optimizer == "official_adam":
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
                amsgrad=True,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=0.001
            )
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
            )
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(
                optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
            )
            scheduler = None
        fit_loader = DataLoader(
            ManifestWaveformDataset(fit_frame, args.data_root),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        validation_loader = DataLoader(
            ManifestWaveformDataset(validation_frame, args.data_root),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        class_weights = None
        if args.class_balanced_loss:
            classes_num = model.backbone.fc.out_features
            counts = np.bincount(
                fit_frame.class_index.to_numpy(), minlength=classes_num
            ).astype(np.float64)
            weights = np.zeros(classes_num, dtype=np.float32)
            present = counts > 0
            weights[present] = 1.0 / counts[present]
            weights[present] *= present.sum() / weights[present].sum()
            class_weights = torch.from_numpy(weights).to(device)

        history = []
        best_score = -float("inf")
        stale_epochs = 0
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                fit_loader,
                task,
                device,
                optimizer,
                args.l0_lambda,
                args.orthogonality_lambda,
                args.max_train_batches,
                class_weights,
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                task,
                device,
                None,
                args.l0_lambda,
                args.orthogonality_lambda,
                args.max_validation_batches,
                None,
            )
            head = model.heads[str(task)]
            active_rank = (
                head.active_rank()
                if isinstance(
                    head,
                    (AdaptiveBoundaryResidual, OrderedAdaptiveBoundaryResidual, SpectralOrderedBoundaryResidual),
                )
                else args.rank
            )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "active_rank": active_rank,
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            score = validation_metrics["macro_accuracy"]
            should_save = args.checkpoint_selection == "last" or score > best_score
            should_stop = False
            if should_save:
                best_score = score
                stale_epochs = 0
                metadata = {
                    "stage": args.stage,
                    "task": task,
                    "seen_tasks": [*previously_seen, task],
                    "method": args.method,
                    "freeze_normalization": args.freeze_normalization,
                    "freeze_boundary": args.freeze_boundary,
                    "head_init_checkpoint": str(args.head_init_checkpoint.resolve())
                    if args.head_init_checkpoint
                    else None,
                    "normalization_init": args.normalization_init,
                    "normalization_checkpoint": str(args.normalization_checkpoint.resolve())
                    if args.normalization_checkpoint
                    else None,
                    "optimizer": args.optimizer,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "workers": args.workers,
                    "learning_rate": args.learning_rate,
                    "boundary_learning_rate": args.boundary_learning_rate,
                    "weight_decay": args.weight_decay,
                    "checkpoint_selection": args.checkpoint_selection,
                    "class_balanced_loss": args.class_balanced_loss,
                    "rank": args.rank,
                    "max_rank": args.max_rank,
                    "initial_rank": args.initial_rank,
                    "alpha": args.alpha,
                    "l0_lambda": args.l0_lambda,
                    "orthogonality_lambda": args.orthogonality_lambda,
                    "seed": args.seed,
                    "manifest": str(args.manifest.resolve()),
                    "best_epoch": epoch,
                    "best_validation_macro_accuracy": best_score,
                    "active_rank": active_rank,
                    "old_domain_hash_before": old_hash_before,
                    "selection_role": args.checkpoint_selection,
                }
                save_checkpoint(args.output, model, optimizer, metadata)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    should_stop = True
            if args.last_output is not None:
                last_metadata = {
                    **metadata,
                    "best_epoch": epoch,
                    "best_validation_macro_accuracy": score,
                    "selection_role": "last",
                }
                save_checkpoint(args.last_output, model, optimizer, last_metadata)
            if should_stop:
                break
            if scheduler is not None:
                scheduler.step()

        best = torch.load(args.output, map_location="cpu", weights_only=False)
        state = best.get("incremental_state_dict", best.get("model_state_dict"))
        if state is None:
            raise RuntimeError("Best checkpoint has no model state")
        load_incremental_state(model, state)
        old_hash_after = tensor_subset_hash(model, old_tasks)
        if old_hash_before != old_hash_after:
            raise RuntimeError("Isolation failure: source/previous-domain state changed")
        router_loader = DataLoader(
            ManifestWaveformDataset(fit_frame, args.data_root),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        current_router_state = collect_rp32_class_prototypes(
            model, router_loader, device, max_batches=args.max_router_batches
        )
        best["metadata"]["old_domain_hash_after"] = old_hash_after
        best["metadata"]["isolation_verified"] = True
        best["history"] = history
        best["router_state"] = {
            **previous_router_state,
            str(task): current_router_state,
        }
        torch.save(best, args.output)
        if args.last_output is not None:
            last = torch.load(args.last_output, map_location="cpu", weights_only=False)
            last_state = last.get("incremental_state_dict", last.get("model_state_dict"))
            if last_state is None:
                raise RuntimeError("Last checkpoint has no model state")
            load_incremental_state(model, last_state)
            if tensor_subset_hash(model, old_tasks) != old_hash_before:
                raise RuntimeError("Isolation failure in final-epoch checkpoint")
            last["metadata"]["old_domain_hash_after"] = old_hash_before
            last["metadata"]["isolation_verified"] = True
            last["history"] = history
            last["router_state"] = {
                **previous_router_state,
                str(task): current_router_state,
            }
            torch.save(last, args.last_output)
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "best_validation_macro_accuracy": best_score,
                    "isolation_verified": True,
                    "device": str(device),
                    "last_checkpoint": str(args.last_output) if args.last_output else None,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
