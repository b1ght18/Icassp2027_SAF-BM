"""Original frozen-path selector; metrics are percentages, budgets are fractions.

Root merging and isolated/persistent ties retain the research implementation's
numerical limitations. See docs/method.md.
"""
from __future__ import annotations
from typing import Any
import numpy as np

def class_metrics(
    labels: np.ndarray,
    source_logits: np.ndarray,
    adapted_logits: np.ndarray,
) -> dict[str, float]:
    source_prediction = source_logits.argmax(axis=1)
    adapted_prediction = adapted_logits.argmax(axis=1)
    source_correct = source_prediction == labels
    adapted_correct = adapted_prediction == labels
    classes = np.unique(labels)

    def balanced(mask: np.ndarray) -> float:
        return 100.0 * float(np.mean([mask[labels == value].mean() for value in classes]))

    source_accuracy = balanced(source_correct)
    adapted_accuracy = balanced(adapted_correct)
    correcting = balanced(~source_correct & adapted_correct)
    harmful = balanced(source_correct & ~adapted_correct)
    return {
        "source_macro_accuracy": source_accuracy,
        "adapted_macro_accuracy": adapted_accuracy,
        "accuracy_gain": adapted_accuracy - source_accuracy,
        "correcting_crossing_rate": correcting,
        "harmful_crossing_rate": harmful,
        "net_correcting_rate": correcting - harmful,
    }


def correctness_events(
    source_logits: np.ndarray,
    endpoint_logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[tuple[float, int, bool]]]:
    """Return exact changes of true-class correctness along an affine path.

    Correctness requires every true-vs-competitor affine margin to be positive.
    The intersection of these one-dimensional halfspaces is an interval, so each
    sample contributes at most an enter and an exit event.
    """
    rows = np.arange(len(labels))
    delta = endpoint_logits - source_logits
    true_source = source_logits[rows, labels]
    true_delta = delta[rows, labels]
    source_correct = source_logits.argmax(axis=1) == labels
    events: list[tuple[float, int, bool]] = []
    tolerance = 1e-12

    for index in range(len(labels)):
        lower, upper = 0.0, 1.0
        possible = True
        for competitor in range(source_logits.shape[1]):
            if competitor == int(labels[index]):
                continue
            intercept = float(true_source[index] - source_logits[index, competitor])
            slope = float(true_delta[index] - delta[index, competitor])
            if slope > tolerance:
                lower = max(lower, -intercept / slope)
            elif slope < -tolerance:
                upper = min(upper, -intercept / slope)
            elif intercept <= 0.0:
                possible = False
                break
        lower = max(0.0, lower)
        upper = min(1.0, upper)
        exists = possible and lower < upper - tolerance
        if bool(source_correct[index]):
            if exists and upper < 1.0 - tolerance:
                events.append((upper, index, False))
        elif exists:
            if lower > tolerance:
                events.append((lower, index, True))
            if upper < 1.0 - tolerance:
                events.append((upper, index, False))
    return source_correct, events


def split_state(
    source_logits: np.ndarray,
    endpoint_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    source_correct, events = correctness_events(source_logits, endpoint_logits, labels)
    class_count = int(source_logits.shape[1])
    totals = np.bincount(labels, minlength=class_count).astype(np.int64)
    present = totals > 0
    correct = np.bincount(labels[source_correct], minlength=class_count).astype(np.int64)
    return {
        "labels": labels,
        "source_correct": source_correct,
        "current": source_correct.copy(),
        "totals": totals,
        "present": present,
        "correct": correct,
        "harmful": np.zeros(class_count, dtype=np.int64),
        "correcting": np.zeros(class_count, dtype=np.int64),
        "events": events,
    }


def update_correctness(state: dict[str, Any], index: int, value: bool) -> None:
    old = bool(state["current"][index])
    if old == value:
        return
    label = int(state["labels"][index])
    state["correct"][label] += 1 if value else -1
    if bool(state["source_correct"][index]):
        state["harmful"][label] += -1 if value else 1
    else:
        state["correcting"][label] += 1 if value else -1
    state["current"][index] = value


def state_metrics(state: dict[str, Any]) -> dict[str, float]:
    present = state["present"]
    totals = state["totals"][present]
    accuracy = 100.0 * float(np.mean(state["correct"][present] / totals))
    harmful = 100.0 * float(np.mean(state["harmful"][present] / totals))
    correcting = 100.0 * float(np.mean(state["correcting"][present] / totals))
    return {
        "macro_accuracy": accuracy,
        "harmful_crossing_rate": harmful,
        "correcting_crossing_rate": correcting,
        "net_correcting_rate": correcting - harmful,
    }


def enumerate_correctness_cells(
    rank: int,
    source_fit: np.ndarray,
    endpoint_fit: np.ndarray,
    fit_labels: np.ndarray,
    source_validation: np.ndarray,
    endpoint_validation: np.ndarray,
    validation_labels: np.ndarray,
) -> list[dict[str, float | int]]:
    fit = split_state(source_fit, endpoint_fit, fit_labels)
    validation = split_state(source_validation, endpoint_validation, validation_labels)
    tagged = [(rho, 0, index, value) for rho, index, value in fit.pop("events")]
    tagged += [(rho, 1, index, value) for rho, index, value in validation.pop("events")]
    tagged.sort(key=lambda row: row[0])
    groups: list[tuple[float, list[tuple[int, int, bool]]]] = []
    for rho, split, index, value in tagged:
        if not groups or abs(rho - groups[-1][0]) > 1e-11:
            groups.append((rho, []))
        groups[-1][1].append((split, index, value))

    cells: list[dict[str, float | int]] = []

    def append_cell(left: float, right: float) -> None:
        fit_values = state_metrics(fit)
        validation_values = state_metrics(validation)
        rho = 0.0 if left == 0.0 else 0.5 * (left + right)
        cells.append(
            {
                "rank": rank,
                "left": left,
                "right": right,
                "radial_scale": rho,
                "fit_macro_accuracy": fit_values["macro_accuracy"],
                "fit_harmful_crossing_rate": fit_values["harmful_crossing_rate"],
                "fit_correcting_crossing_rate": fit_values["correcting_crossing_rate"],
                "fit_net_correcting_rate": fit_values["net_correcting_rate"],
                "validation_macro_accuracy": validation_values["macro_accuracy"],
                "harmful_crossing_rate": validation_values["harmful_crossing_rate"],
                "correcting_crossing_rate": validation_values["correcting_crossing_rate"],
                "net_correcting_rate": validation_values["net_correcting_rate"],
            }
        )

    first = groups[0][0] if groups else 1.0
    append_cell(0.0, first)
    for group_index, (rho, events) in enumerate(groups):
        for split, index, value in events:
            update_correctness(fit if split == 0 else validation, index, value)
        right = groups[group_index + 1][0] if group_index + 1 < len(groups) else 1.0
        if right - rho > 1e-12:
            append_cell(rho, right)
    return cells


def feasible(
    cell: dict[str, float | int], fit_budget: float, validation_budget: float
) -> bool:
    return (
        float(cell["fit_harmful_crossing_rate"]) <= 100.0 * fit_budget + 1e-10
        and float(cell["harmful_crossing_rate"])
        <= 100.0 * validation_budget + 1e-10
    )


def cell_score(cell: dict[str, float | int]) -> tuple[float, ...]:
    return (
        float(cell["validation_macro_accuracy"]),
        float(cell["net_correcting_rate"]),
        -float(cell["harmful_crossing_rate"]),
        -float(cell["radial_scale"]),
    )


def select_fixed_path(
    cells: list[dict[str, float | int]],
    fit_budget: float,
    validation_budget: float,
) -> dict[str, float | int]:
    candidates = [cell for cell in cells if feasible(cell, fit_budget, validation_budget)]
    if not candidates:
        raise RuntimeError("source-anchored path lost feasibility")
    return dict(max(candidates, key=cell_score))


def select_rank_family(
    cells: list[dict[str, float | int]],
    fit_budget: float,
    validation_budget: float,
    rank_tolerance: float,
) -> dict[str, float | int]:
    candidates = [cell for cell in cells if feasible(cell, fit_budget, validation_budget)]
    if not candidates:
        raise RuntimeError("source anchor disappeared from the rank family")
    best_accuracy = max(float(cell["validation_macro_accuracy"]) for cell in candidates)
    useful = [
        cell
        for cell in candidates
        if float(cell["validation_macro_accuracy"]) >= best_accuracy - rank_tolerance
    ]
    minimum_rank = min(int(cell["rank"]) for cell in useful)
    selected = max(
        (cell for cell in useful if int(cell["rank"]) == minimum_rank),
        key=cell_score,
    )
    output = dict(selected)
    output["best_feasible_validation_accuracy"] = best_accuracy
    return output


def gauge_weight(weight: np.ndarray) -> np.ndarray:
    return weight - weight.mean(axis=0, keepdims=True)


def gauge_bias(bias: np.ndarray) -> np.ndarray:
    return bias - bias.mean()


def matched_random_residual(
    delta_weight: np.ndarray,
    delta_bias: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Randomize singular directions while exactly preserving nonzero spectrum."""
    rng = np.random.default_rng(seed)
    centered_weight = gauge_weight(delta_weight.astype(np.float64))
    centered_bias = gauge_bias(delta_bias.astype(np.float64))
    _, singular, _ = np.linalg.svd(centered_weight, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros_like(centered_weight), np.zeros_like(centered_bias), {
            "effective_rank": 0,
            "spectrum_max_abs_error": 0.0,
            "frobenius_abs_error": 0.0,
            "bias_norm_abs_error": 0.0,
        }
    effective_rank = int(np.sum(singular > singular[0] * 1e-7))
    singular = singular[:effective_rank]
    classes, feature_dim = centered_weight.shape

    left_random = rng.standard_normal((classes, effective_rank))
    left_random -= left_random.mean(axis=0, keepdims=True)
    left, _ = np.linalg.qr(left_random, mode="reduced")
    right_random = rng.standard_normal((feature_dim, effective_rank))
    right, _ = np.linalg.qr(right_random, mode="reduced")
    randomized_weight = (left[:, :effective_rank] * singular) @ right[:, :effective_rank].T
    randomized_weight = gauge_weight(randomized_weight)

    bias_norm = float(np.linalg.norm(centered_bias))
    if bias_norm:
        randomized_bias = rng.standard_normal(classes)
        randomized_bias -= randomized_bias.mean()
        randomized_bias *= bias_norm / max(float(np.linalg.norm(randomized_bias)), 1e-20)
    else:
        randomized_bias = np.zeros_like(centered_bias)

    randomized_singular = np.linalg.svd(randomized_weight, compute_uv=False)[:effective_rank]
    audit = {
        "effective_rank": effective_rank,
        "spectrum_max_abs_error": float(np.max(np.abs(randomized_singular - singular))),
        "frobenius_abs_error": abs(
            float(np.linalg.norm(randomized_weight)) - float(np.linalg.norm(centered_weight))
        ),
        "bias_norm_abs_error": abs(float(np.linalg.norm(randomized_bias)) - bias_norm),
    }
    return randomized_weight, randomized_bias, audit


def endpoint_logits(
    features: np.ndarray,
    source_logits: np.ndarray,
    delta_weight: np.ndarray,
    delta_bias: np.ndarray,
) -> np.ndarray:
    return source_logits + features @ delta_weight.T + delta_bias


def exact_path_selection(
    rank: int,
    delta_weight: np.ndarray,
    delta_bias: np.ndarray,
    arrays: dict[str, np.ndarray],
    fit_budget: float,
    validation_budget: float,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    endpoint_fit = endpoint_logits(
        arrays["fit_features"], arrays["source_fit"], delta_weight, delta_bias
    )
    endpoint_validation = endpoint_logits(
        arrays["validation_features"],
        arrays["source_validation"],
        delta_weight,
        delta_bias,
    )
    cells = enumerate_correctness_cells(
        rank,
        arrays["source_fit"],
        endpoint_fit,
        arrays["fit_labels"],
        arrays["source_validation"],
        endpoint_validation,
        arrays["validation_labels"],
    )
    return select_fixed_path(cells, fit_budget, validation_budget), cells

