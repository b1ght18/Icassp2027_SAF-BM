from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BoundaryResidualRegularization:
    expected_active_components: torch.Tensor
    orthogonality: torch.Tensor


def fix_softmax_gauge(
    weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Remove class-common logits without changing softmax or argmax decisions."""
    centered_weight = weight - weight.mean(dim=0, keepdim=True)
    if bias is None:
        return centered_weight
    return centered_weight, bias - bias.mean()


class AdaptiveBoundaryResidual(nn.Module):
    """Direct source-anchored boundary migration with learnable spectral gates.

    The effective classifier is always W_t = W_0 + Delta W_t. Normalization is
    handled by the backbone; this module changes only the decision boundary.
    """

    def __init__(
        self,
        source_weight: torch.Tensor,
        source_bias: torch.Tensor,
        max_rank: int | None = None,
        alpha: float = 8.0,
        temperature: float = 2.0 / 3.0,
        gate_low: float = -0.1,
        gate_high: float = 1.1,
        initial_log_alpha: float = 2.0,
    ):
        super().__init__()
        classes, feature_dim = source_weight.shape
        max_rank = min(classes, feature_dim) if max_rank is None else max_rank
        if not 1 <= max_rank <= min(classes, feature_dim):
            raise ValueError(f"max_rank must be in [1, {min(classes, feature_dim)}]")
        self.register_buffer("source_weight", source_weight.detach().clone())
        self.register_buffer("source_bias", source_bias.detach().clone())
        self.max_rank = max_rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / max_rank
        self.temperature = float(temperature)
        self.gate_low = float(gate_low)
        self.gate_high = float(gate_high)
        self.A = nn.Parameter(torch.empty(max_rank, feature_dim))
        self.B = nn.Parameter(torch.zeros(classes, max_rank))
        self.log_alpha = nn.Parameter(torch.full((max_rank,), float(initial_log_alpha)))
        self.delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        nn.init.normal_(self.A, mean=0.0, std=0.01)

    def gate_probabilities(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.log_alpha) * (self.gate_high - self.gate_low) + self.gate_low
        return stretched.clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        offset = self.temperature * torch.log(
            torch.tensor(-self.gate_low / self.gate_high, device=self.log_alpha.device)
        )
        return torch.sigmoid(self.log_alpha - offset).sum()

    def gates(self, stochastic: bool | None = None, hard: bool = True) -> torch.Tensor:
        stochastic = self.training if stochastic is None else stochastic
        if stochastic:
            uniform = torch.rand_like(self.log_alpha).clamp(1e-6, 1.0 - 1e-6)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            soft = torch.sigmoid((logistic + self.log_alpha) / self.temperature)
            soft = (soft * (self.gate_high - self.gate_low) + self.gate_low).clamp(0.0, 1.0)
        else:
            soft = self.gate_probabilities()
        if not hard:
            return soft
        binary = (soft >= 0.5).to(soft.dtype)
        if stochastic:
            return binary.detach() - soft.detach() + soft
        return binary

    def delta_weight(self, stochastic: bool | None = None, hard: bool = True) -> torch.Tensor:
        gates = self.gates(stochastic=stochastic, hard=hard)
        return self.scaling * ((self.B * gates.unsqueeze(0)) @ self.A)

    def effective_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_weight + self.delta_weight(), self.source_bias + self.delta_bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight, bias = self.effective_parameters()
        return F.linear(features, weight, bias)

    def regularization(self) -> BoundaryResidualRegularization:
        normalized_a = F.normalize(self.A, dim=1)
        normalized_b = F.normalize(self.B, dim=0)
        identity = torch.eye(self.max_rank, device=self.A.device, dtype=self.A.dtype)
        gram_a = normalized_a @ normalized_a.T
        gram_b = normalized_b.T @ normalized_b
        orthogonality = (gram_a - identity).square().mean() + (gram_b - identity).square().mean()
        return BoundaryResidualRegularization(self.expected_l0(), orthogonality)

    def active_rank(self, threshold: float = 0.5) -> int:
        return int((self.gate_probabilities() >= threshold).sum().item())

    def compact_state(self, threshold: float = 0.5) -> dict[str, torch.Tensor | int | float]:
        active = self.gate_probabilities() >= threshold
        if not bool(active.any()):
            strongest = int(self.gate_probabilities().argmax().item())
            active[strongest] = True
        return {
            "source_weight": self.source_weight.detach().cpu(),
            "source_bias": self.source_bias.detach().cpu(),
            "A": self.A[active].detach().cpu(),
            "B": self.B[:, active].detach().cpu(),
            "delta_bias": self.delta_bias.detach().cpu(),
            "rank": int(active.sum().item()),
            "scaling": self.scaling,
            "gate_probabilities": self.gate_probabilities()[active].detach().cpu(),
        }


class FixedRankBoundaryResidual(nn.Module):
    def __init__(
        self,
        source_weight: torch.Tensor,
        source_bias: torch.Tensor,
        rank: int,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.register_buffer("source_weight", source_weight.detach().clone())
        self.register_buffer("source_bias", source_bias.detach().clone())
        self.A = nn.Parameter(torch.empty(rank, source_weight.shape[1]))
        self.B = nn.Parameter(torch.zeros(source_weight.shape[0], rank))
        self.delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        self.scaling = float(alpha) / rank
        nn.init.normal_(self.A, mean=0.0, std=0.01)

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.B @ self.A)

    def effective_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_weight + self.delta_weight(), self.source_bias + self.delta_bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight, bias = self.effective_parameters()
        return F.linear(features, weight, bias)


class NestedRankBoundaryResidual(nn.Module):
    """One-shot ordered LoRA prefixes for source-anchored boundaries.

    A single maximum-rank factorization exposes every prefix rank ``1..R``.
    The class factor and bias are centered so the learned residual is in the
    softmax decision gauge by construction.  ``component_scale`` is independent
    of ``max_rank``; consequently extending a rank-3 experiment to rank 9 does
    not silently reduce the update scale of the first three atoms.
    """

    def __init__(
        self,
        source_weight: torch.Tensor,
        source_bias: torch.Tensor,
        max_rank: int,
        component_scale: float = 8.0 / 3.0,
    ):
        super().__init__()
        classes, feature_dim = source_weight.shape
        decision_rank = min(classes - 1, feature_dim)
        if not 1 <= max_rank <= decision_rank:
            raise ValueError(
                f"max_rank must be in [1, {decision_rank}] after softmax-gauge removal"
            )
        self.register_buffer("source_weight", source_weight.detach().clone())
        self.register_buffer("source_bias", source_bias.detach().clone())
        self.max_rank = int(max_rank)
        self.component_scale = float(component_scale)
        self.A = nn.Parameter(torch.empty(max_rank, feature_dim))
        self.B = nn.Parameter(torch.zeros(classes, max_rank))
        self.delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        nn.init.normal_(self.A, mean=0.0, std=0.01)

    def _check_rank(self, rank: int) -> int:
        rank = int(rank)
        if not 0 <= rank <= self.max_rank:
            raise ValueError(f"rank must be in [0, {self.max_rank}]")
        return rank

    def centered_class_factor(self) -> torch.Tensor:
        return self.B - self.B.mean(dim=0, keepdim=True)

    def centered_delta_bias(self) -> torch.Tensor:
        return self.delta_bias - self.delta_bias.mean()

    def delta_weight(self, rank: int | None = None) -> torch.Tensor:
        rank = self.max_rank if rank is None else self._check_rank(rank)
        if rank == 0:
            return torch.zeros_like(self.source_weight)
        return self.component_scale * (
            self.centered_class_factor()[:, :rank] @ self.A[:rank]
        )

    def effective_parameters(
        self, rank: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = self.max_rank if rank is None else self._check_rank(rank)
        if rank == 0:
            return self.source_weight, self.source_bias
        return (
            self.source_weight + self.delta_weight(rank),
            self.source_bias + self.centered_delta_bias(),
        )

    def forward(self, features: torch.Tensor, rank: int | None = None) -> torch.Tensor:
        weight, bias = self.effective_parameters(rank)
        return F.linear(features, weight, bias)


class OrderedAdaptiveBoundaryResidual(nn.Module):
    """Direct boundary residual with a learnable ordered effective-rank cutoff.

    Orthogonality regularization makes factor components spectral-like, while a
    monotone gate ensures that component j+1 cannot remain active after component j
    is removed. This breaks the permutation symmetry of independent L0 gates.
    """

    def __init__(
        self,
        source_weight: torch.Tensor,
        source_bias: torch.Tensor,
        max_rank: int | None = None,
        alpha: float = 8.0,
        temperature: float = 0.5,
        initial_rank: float | None = None,
        minimum_rank: int = 1,
    ):
        super().__init__()
        classes, feature_dim = source_weight.shape
        max_rank = min(classes, feature_dim) if max_rank is None else max_rank
        if not 1 <= minimum_rank <= max_rank <= min(classes, feature_dim):
            raise ValueError("Require 1 <= minimum_rank <= max_rank <= matrix rank")
        self.register_buffer("source_weight", source_weight.detach().clone())
        self.register_buffer("source_bias", source_bias.detach().clone())
        self.max_rank = int(max_rank)
        self.minimum_rank = int(minimum_rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.max_rank
        self.temperature = float(temperature)
        self.A = nn.Parameter(torch.empty(max_rank, feature_dim))
        self.B = nn.Parameter(torch.zeros(classes, max_rank))
        self.delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        nn.init.normal_(self.A, mean=0.0, std=0.01)
        if max_rank == minimum_rank:
            initial_fraction = 0.5
        else:
            initial_rank = max_rank - 0.5 if initial_rank is None else initial_rank
            if not minimum_rank < initial_rank < max_rank:
                raise ValueError("initial_rank must be strictly between minimum_rank and max_rank")
            initial_fraction = (initial_rank - minimum_rank) / (max_rank - minimum_rank)
        initial_fraction = min(max(initial_fraction, 1e-4), 1.0 - 1e-4)
        initial_logit = torch.logit(torch.tensor(initial_fraction)).item()
        self.rank_logit = nn.Parameter(torch.tensor(initial_logit))

    def continuous_rank(self) -> torch.Tensor:
        if self.max_rank == self.minimum_rank:
            return self.rank_logit * 0.0 + float(self.max_rank)
        return self.minimum_rank + (self.max_rank - self.minimum_rank) * torch.sigmoid(
            self.rank_logit
        )

    def gate_probabilities(self) -> torch.Tensor:
        centers = torch.arange(
            self.max_rank, device=self.rank_logit.device, dtype=self.rank_logit.dtype
        ) + 0.5
        return torch.sigmoid((self.continuous_rank() - centers) / self.temperature)

    def expected_l0(self) -> torch.Tensor:
        return self.gate_probabilities().sum()

    def gates(self, stochastic: bool | None = None, hard: bool = True) -> torch.Tensor:
        del stochastic  # Ordered cutoff is deterministic and reproducible.
        soft = self.gate_probabilities()
        if not hard:
            return soft
        binary = (soft >= 0.5).to(soft.dtype)
        if self.training:
            return binary.detach() - soft.detach() + soft
        return binary

    def delta_weight(self, stochastic: bool | None = None, hard: bool = True) -> torch.Tensor:
        gates = self.gates(stochastic=stochastic, hard=hard)
        return self.scaling * ((self.B * gates.unsqueeze(0)) @ self.A)

    def effective_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_weight + self.delta_weight(), self.source_bias + self.delta_bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight, bias = self.effective_parameters()
        return F.linear(features, weight, bias)

    def regularization(self) -> BoundaryResidualRegularization:
        normalized_a = F.normalize(self.A, dim=1)
        normalized_b = F.normalize(self.B, dim=0)
        identity = torch.eye(self.max_rank, device=self.A.device, dtype=self.A.dtype)
        gram_a = normalized_a @ normalized_a.T
        gram_b = normalized_b.T @ normalized_b
        orthogonality = (gram_a - identity).square().mean() + (
            gram_b - identity
        ).square().mean()
        return BoundaryResidualRegularization(self.expected_l0(), orthogonality)

    def active_rank(self, threshold: float = 0.5) -> int:
        return int((self.gate_probabilities() >= threshold).sum().item())

    def compact_state(self, threshold: float = 0.5) -> dict[str, torch.Tensor | int | float]:
        rank = max(self.minimum_rank, self.active_rank(threshold))
        return {
            "source_weight": self.source_weight.detach().cpu(),
            "source_bias": self.source_bias.detach().cpu(),
            "A": self.A[:rank].detach().cpu(),
            "B": self.B[:, :rank].detach().cpu(),
            "delta_bias": self.delta_bias.detach().cpu(),
            "rank": rank,
            "scaling": self.scaling,
            "continuous_rank": float(self.continuous_rank().detach().cpu()),
            "gate_probabilities": self.gate_probabilities()[:rank].detach().cpu(),
        }


class SpectralOrderedBoundaryResidual(nn.Module):
    """Source-anchored residual with explicit orthonormal spectral factors."""

    def __init__(
        self,
        source_weight: torch.Tensor,
        source_bias: torch.Tensor,
        max_rank: int | None = None,
        alpha: float = 8.0,
        temperature: float = 0.5,
        initial_rank: float | None = None,
        minimum_rank: int = 1,
    ):
        super().__init__()
        classes, feature_dim = source_weight.shape
        max_rank = min(classes, feature_dim) if max_rank is None else max_rank
        if not 1 <= minimum_rank <= max_rank <= min(classes, feature_dim):
            raise ValueError("Invalid spectral rank range")
        self.register_buffer("source_weight", source_weight.detach().clone())
        self.register_buffer("source_bias", source_bias.detach().clone())
        self.max_rank = int(max_rank)
        self.minimum_rank = int(minimum_rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.max_rank
        self.temperature = float(temperature)
        self.U_raw = nn.Parameter(torch.randn(classes, max_rank) * 0.02)
        self.V_raw = nn.Parameter(torch.randn(feature_dim, max_rank) * 0.02)
        self.amplitudes = nn.Parameter(torch.zeros(max_rank))
        self.delta_bias = nn.Parameter(torch.zeros_like(source_bias))
        initial_rank = max_rank - 0.5 if initial_rank is None else initial_rank
        if max_rank == minimum_rank:
            fraction = 0.5
        else:
            if not minimum_rank < initial_rank < max_rank:
                raise ValueError("initial_rank must be strictly inside rank range")
            fraction = (initial_rank - minimum_rank) / (max_rank - minimum_rank)
        fraction = min(max(fraction, 1e-4), 1.0 - 1e-4)
        self.rank_logit = nn.Parameter(torch.logit(torch.tensor(fraction)))

    def continuous_rank(self) -> torch.Tensor:
        if self.max_rank == self.minimum_rank:
            return self.rank_logit * 0.0 + float(self.max_rank)
        return self.minimum_rank + (self.max_rank - self.minimum_rank) * torch.sigmoid(
            self.rank_logit
        )

    def gate_probabilities(self) -> torch.Tensor:
        centers = torch.arange(
            self.max_rank, device=self.rank_logit.device, dtype=self.rank_logit.dtype
        ) + 0.5
        return torch.sigmoid((self.continuous_rank() - centers) / self.temperature)

    def expected_l0(self) -> torch.Tensor:
        return self.gate_probabilities().sum()

    def gates(self, stochastic: bool | None = None, hard: bool = True) -> torch.Tensor:
        del stochastic
        soft = self.gate_probabilities()
        if not hard:
            return soft
        binary = (soft >= 0.5).to(soft.dtype)
        if self.training:
            return binary.detach() - soft.detach() + soft
        return binary

    def orthonormal_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        left, _ = torch.linalg.qr(self.U_raw, mode="reduced")
        right, _ = torch.linalg.qr(self.V_raw, mode="reduced")
        return left, right

    def delta_weight(self, stochastic: bool | None = None, hard: bool = True) -> torch.Tensor:
        left, right = self.orthonormal_factors()
        strengths = self.scaling * self.amplitudes * self.gates(stochastic, hard)
        return (left * strengths.unsqueeze(0)) @ right.T

    def effective_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_weight + self.delta_weight(), self.source_bias + self.delta_bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight, bias = self.effective_parameters()
        return F.linear(features, weight, bias)

    def regularization(self) -> BoundaryResidualRegularization:
        magnitudes = self.amplitudes.abs()
        ordering = F.relu(magnitudes[1:] - magnitudes[:-1]).square().mean()
        return BoundaryResidualRegularization(self.expected_l0(), ordering)

    def active_rank(self, threshold: float = 0.5) -> int:
        return int((self.gate_probabilities() >= threshold).sum().item())

    def compact_state(self, threshold: float = 0.5) -> dict[str, torch.Tensor | int | float]:
        rank = max(self.minimum_rank, self.active_rank(threshold))
        left, right = self.orthonormal_factors()
        return {
            "source_weight": self.source_weight.detach().cpu(),
            "source_bias": self.source_bias.detach().cpu(),
            "U": left[:, :rank].detach().cpu(),
            "V": right[:, :rank].detach().cpu(),
            "amplitudes": (self.scaling * self.amplitudes[:rank]).detach().cpu(),
            "delta_bias": self.delta_bias.detach().cpu(),
            "rank": rank,
            "continuous_rank": float(self.continuous_rank().detach().cpu()),
            "gate_probabilities": self.gate_probabilities()[:rank].detach().cpu(),
        }
