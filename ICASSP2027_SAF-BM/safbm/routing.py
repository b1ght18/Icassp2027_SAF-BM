"""Minimum-entropy routing over explicitly provided seen branches."""
from __future__ import annotations
from typing import Iterable
import torch

def entropy_route(
    logits_by_domain: dict[int, torch.Tensor], candidate_domains: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = tuple(int(domain) for domain in candidate_domains)
    stacked = torch.stack([logits_by_domain[domain] for domain in candidates], dim=1)
    entropies = -(stacked.softmax(-1) * stacked.log_softmax(-1)).sum(-1)
    route_indices = entropies.argmin(dim=1)
    rows = torch.arange(len(route_indices), device=stacked.device)
    chosen_logits = stacked[rows, route_indices]
    routes = torch.as_tensor(candidates, device=stacked.device)[route_indices]
    return chosen_logits, routes



def sequential_metrics(logits, labels, domains):
    """logits[b,n,c]: frozen branch b on ALL held-out examples, in shared ID order.

    Outputs percentages/percentage points; unavailable domains are None.
    Each arrival can change routing although all earlier branches remain fixed.
    """
    import numpy as np
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels, domains = np.asarray(labels), np.asarray(domains)
    if logits.ndim != 3 or logits.shape[1] != len(labels) or labels.shape != domains.shape:
        raise ValueError('Expected logits [branches,samples,classes], labels/domains [samples]')
    if not torch.isfinite(logits).all() or not len(labels):
        raise ValueError('Empty or nonfinite logits')
    count = logits.shape[0]
    if domains.min() < 0 or domains.max() >= count or labels.min() < 0 or labels.max() >= logits.shape[2]:
        raise ValueError('Invalid class/domain indices')
    matrix = np.full((count, count), np.nan)
    route_counts = []
    by_domain = {i: logits[i] for i in range(count)}
    for t in range(count):
        chosen, routes = entropy_route(by_domain, range(t + 1))
        correct = chosen.argmax(-1).numpy() == labels
        for j in range(t + 1):
            mask = domains == j
            if mask.any():
                matrix[t, j] = 100 * np.mean([correct[mask & (labels == c)].mean()
                                            for c in np.unique(labels[mask])])
        seen = domains <= t
        route_counts.append(np.bincount(routes.numpy()[seen], minlength=count).tolist())
    prior = [j for j in range(count - 1) if np.isfinite(matrix[j, j])]
    forgetting = [float(np.nanmax(matrix[j:count-1, j]) - matrix[-1, j]) for j in prior]
    bwt = [float(matrix[-1, j] - matrix[j, j]) for j in prior]
    return {'accuracy_matrix':[[float(v) if np.isfinite(v) else None for v in row] for row in matrix],
        'final_avg':float(np.nanmean(matrix[-1])),
        'forgetting':float(np.mean(forgetting)) if prior else None,
        'backward_transfer':float(np.mean(bwt)) if prior else None,
        'route_counts_by_arrival':route_counts,
        'units':'accuracy in percent; forgetting/backward transfer in percentage points'}
