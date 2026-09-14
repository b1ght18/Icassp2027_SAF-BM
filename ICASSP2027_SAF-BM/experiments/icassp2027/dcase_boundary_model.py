from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .boundary_residual import (
        AdaptiveBoundaryResidual,
        FixedRankBoundaryResidual,
        OrderedAdaptiveBoundaryResidual,
        SpectralOrderedBoundaryResidual,
    )
except ImportError:
    from boundary_residual import (
        AdaptiveBoundaryResidual,
        FixedRankBoundaryResidual,
        OrderedAdaptiveBoundaryResidual,
        SpectralOrderedBoundaryResidual,
    )


from safbm.backbone import MCnn14


def make_backbone(classes_num: int = 10, nb_tasks: int = 3) -> MCnn14:
    return MCnn14(
        sample_rate=32_000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14_000,
        classes_num=classes_num,
        nb_tasks=nb_tasks,
    )


def unwrap_state(raw: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = raw.get(key)
        if isinstance(candidate, dict):
            return candidate
    return raw  # type: ignore[return-value]


def infer_backbone_shape(
    state: dict[str, torch.Tensor],
    classes_num: int | None,
    nb_tasks: int | None,
) -> tuple[int, int]:
    if classes_num is None:
        candidates = [
            value
            for name, value in state.items()
            if name.removeprefix("backbone.") == "fc.weight"
        ]
        if len(candidates) != 1:
            raise RuntimeError("Cannot infer classifier size from source checkpoint")
        classes_num = int(candidates[0].shape[0])
    if nb_tasks is None:
        indices = []
        for name in state:
            cleaned = name.removeprefix("backbone.")
            match = re.search(r"(?:^|\.)(?:bn0|bnF|bnS)\.(\d+)\.", cleaned)
            if match:
                indices.append(int(match.group(1)))
        nb_tasks = max(indices) + 1 if indices else 3
    return int(classes_num), int(nb_tasks)


def load_source_backbone(
    path: Path,
    classes_num: int | None = None,
    nb_tasks: int | None = None,
) -> MCnn14:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = unwrap_state(raw)
    classes_num, nb_tasks = infer_backbone_shape(state, classes_num, nb_tasks)
    model = make_backbone(classes_num=classes_num, nb_tasks=nb_tasks)
    cleaned = {}
    for name, value in state.items():
        if name.startswith("backbone."):
            cleaned[name.removeprefix("backbone.")] = value
        elif name in model.state_dict():
            cleaned[name] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    material_missing = [name for name in missing if not name.startswith(("bn0.1", "bn0.2"))]
    if material_missing or unexpected:
        raise RuntimeError(
            f"Source checkpoint mismatch: missing={material_missing[:10]}, unexpected={unexpected[:10]}"
        )
    return model


def copy_normalization_branch(model: MCnn14, source: int, target: int) -> None:
    branch_lists = [model.bn0]
    for block in (
        model.conv_block1,
        model.conv_block2,
        model.conv_block3,
        model.conv_block4,
        model.conv_block5,
        model.conv_block6,
    ):
        branch_lists.extend((block.bnF, block.bnS))
    for branches in branch_lists:
        branches[target].load_state_dict(branches[source].state_dict())


class BoundaryMigrationModel(nn.Module):
    """Frozen acoustic encoder plus isolated domain-normalization and boundary shifts.

    For every target domain t, logits are computed with W_t = W_0 + Delta W_t.
    Adaptive gating changes the effective rank of Delta W_t; it never reconstructs
    features and never changes the source anchor W_0.
    """

    def __init__(
        self,
        backbone: MCnn14,
        method: str,
        rank: int = 4,
        max_rank: int = 10,
        alpha: float = 8.0,
        initial_log_alpha: float = 2.0,
        initial_rank: float | None = None,
        task_ranks: dict[int, int] | None = None,
    ):
        super().__init__()
        if method not in {"bn_only", "fixed", "adaptive", "adaptive_ordered", "spectral_ordered"}:
            raise ValueError(f"Unknown residual method: {method}")
        self.backbone = backbone
        self.method = method
        self.adapt_normalization = True
        self.heads = nn.ModuleDict()
        for task in range(1, len(backbone.bn0)):
            task_rank = rank if task_ranks is None else int(task_ranks.get(task, rank))
            if method in {"bn_only", "fixed"}:
                head = FixedRankBoundaryResidual(
                    backbone.fc.weight, backbone.fc.bias, task_rank, alpha
                )
            elif method == "adaptive":
                head = AdaptiveBoundaryResidual(
                    backbone.fc.weight,
                    backbone.fc.bias,
                    max_rank=max_rank,
                    alpha=alpha,
                    initial_log_alpha=initial_log_alpha,
                )
            elif method == "adaptive_ordered":
                head = OrderedAdaptiveBoundaryResidual(
                    backbone.fc.weight,
                    backbone.fc.bias,
                    max_rank=max_rank,
                    alpha=alpha,
                    initial_rank=initial_rank,
                )
            else:
                head = SpectralOrderedBoundaryResidual(
                    backbone.fc.weight,
                    backbone.fc.bias,
                    max_rank=max_rank,
                    alpha=alpha,
                    initial_rank=initial_rank,
                )
            self.heads[str(task)] = head
        self.freeze_all()

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def activate_stage(
        self,
        task: int,
        freeze_normalization: bool = False,
        freeze_boundary: bool = False,
    ) -> None:
        if str(task) not in self.heads:
            raise ValueError(f"Task {task} has no domain-specific head")
        self.freeze_all()
        self.adapt_normalization = not freeze_normalization
        if not freeze_normalization:
            for parameter in self._normalization_modules(task).parameters():
                parameter.requires_grad = True
        if self.method != "bn_only" and not freeze_boundary:
            for parameter in self.heads[str(task)].parameters():
                parameter.requires_grad = True

    def _normalization_modules(self, task: int) -> nn.ModuleList:
        modules: list[nn.Module] = [self.backbone.bn0[task]]
        for block in (
            self.backbone.conv_block1,
            self.backbone.conv_block2,
            self.backbone.conv_block3,
            self.backbone.conv_block4,
            self.backbone.conv_block5,
            self.backbone.conv_block6,
        ):
            modules.extend((block.bnF[task], block.bnS[task]))
        return nn.ModuleList(modules)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen branches are always inference-only. The current branch is explicitly
        # switched by set_stage_mode, preventing accidental running-stat updates.
        for task in range(len(self.backbone.bn0)):
            self._normalization_modules(task).eval()
        return self

    def set_stage_mode(self, task: int, training: bool) -> None:
        # Residual-only phase uses deterministic frozen features: no BN-stat updates
        # and no backbone dropout. Joint/BN phase retains the original augmentation.
        self.train(training and self.adapt_normalization)
        self._normalization_modules(task).train(training and self.adapt_normalization)
        self.heads[str(task)].train(training)

    def features(self, waveforms: torch.Tensor, task: int) -> torch.Tensor:
        x = self.backbone.spectrogram_extractor(waveforms)
        x = self.backbone.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.backbone.bn0[task](x)
        x = x.transpose(1, 3)
        for block in (
            self.backbone.conv_block1,
            self.backbone.conv_block2,
            self.backbone.conv_block3,
            self.backbone.conv_block4,
            self.backbone.conv_block5,
            self.backbone.conv_block6,
        ):
            x = block(x, pool_size=(2, 2), pool_type="avg", task=task)
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        maximum, _ = torch.max(x, dim=2)
        return maximum + torch.mean(x, dim=2)

    def forward(self, waveforms: torch.Tensor, task: int) -> torch.Tensor:
        features = self.features(waveforms, task)
        if task == 0:
            return F.linear(features, self.backbone.fc.weight, self.backbone.fc.bias)
        return self.heads[str(task)](features)


def tensor_subset_hash(model: BoundaryMigrationModel, tasks: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    prefixes = ["backbone.fc."]
    for task in tasks:
        prefixes.extend((f"heads.{task}.",))
    for name, value in sorted(model.state_dict().items()):
        is_selected_head = any(name.startswith(prefix) for prefix in prefixes)
        is_selected_bn = any(
            token in name
            for task in tasks
            for token in (f"bn0.{task}.", f"bnF.{task}.", f"bnS.{task}.")
        )
        if is_selected_head or is_selected_bn:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def incremental_state_dict(
    model: BoundaryMigrationModel, tasks: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Serialize only domain increments; the frozen D1 checkpoint remains the anchor."""
    selected: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        target_bn = any(
            token in name
            for task in tasks
            for token in (f"bn0.{task}.", f"bnF.{task}.", f"bnS.{task}.")
        )
        target_head = model.method != "bn_only" and any(
            name.startswith(f"heads.{task}.") for task in tasks
        )
        source_copy = name.endswith(("source_weight", "source_bias"))
        if target_bn or (target_head and not source_copy):
            selected[name] = value.detach().cpu().clone()
    if not selected:
        raise RuntimeError("Incremental checkpoint would be empty")
    return selected


def load_incremental_state(
    model: BoundaryMigrationModel, state: dict[str, torch.Tensor]
) -> None:
    expected = set(model.state_dict())
    unexpected = sorted(set(state) - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected incremental checkpoint keys: {unexpected[:10]}")
    model.load_state_dict(state, strict=False)
