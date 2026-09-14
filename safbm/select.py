"""Joint selection of a frozen raw rank endpoint and its deployed fraction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.icassp2027.train_cached_hcrg import RankGrowingBoundaryResidual
from experiments.icassp2027.boundary_residual import fix_softmax_gauge
from .events import exact_path_selection, select_rank_family, class_metrics, gauge_bias
from .io import load_fit_validation, digest, write_json


def raw_residuals(proposal, arrays):
    """Retain research arithmetic and select RAW endpoints, not restored states."""
    w = torch.from_numpy(arrays['source_weight'].astype(np.float32))
    b = torch.from_numpy(arrays['source_bias'].astype(np.float32))
    maximum, alpha = int(proposal['max_rank']), float(proposal['alpha'])
    if not np.isclose(proposal['factor_scaling'], alpha / maximum, rtol=0, atol=1e-12):
        raise ValueError('Inconsistent factor scaling')
    for name, expected in [('source_weight', w), ('source_bias', b)]:
        if not torch.equal(proposal[name], expected):
            raise ValueError(f'Cache/proposal {name} mismatch')
    states = proposal['candidate_raw_states']
    if not set(range(1, maximum + 1)).issubset({int(r) for r in states}):
        raise ValueError('Incomplete frozen rank family')
    result = {0: (np.zeros_like(arrays['source_weight']), np.zeros_like(arrays['source_bias']))}
    for key, state in states.items():
        rank = int(key)
        if rank == 0:
            continue
        if not 1 <= rank <= maximum:
            raise ValueError('Invalid candidate rank')
        if not torch.equal(state['source_weight'], w) or not torch.equal(state['source_bias'], b):
            raise ValueError('Candidate changed the source anchor')
        model = RankGrowingBoundaryResidual(w, b, maximum, alpha)
        model.active_rank = rank
        model.load_state_dict(state)
        result[rank] = (fix_softmax_gauge(model.delta_weight().detach()).double().numpy(),
                        gauge_bias(model.delta_bias.detach().double().numpy()))
    return result


def select(proposal_path, cache_path, cfg, output_dir):
    output_dir = Path(output_dir)
    if (output_dir / 'head.npz').exists():
        raise FileExistsError('A frozen head already exists; choose a new output directory')
    arrays = load_fit_validation(cache_path)
    proposal = torch.load(proposal_path, map_location='cpu', weights_only=False)
    if any(int(proposal[key]) != int(arrays[key]) for key in ['task', 'stage']):
        raise ValueError('Cache/proposal domain or stage mismatch')
    run_path = Path(proposal_path).with_name('run.json')
    if run_path.exists():
        provenance = json.loads(run_path.read_text())
        if provenance['fit_validation_sha256'] != digest(cache_path) or provenance['proposal_sha256'] != digest(proposal_path):
            raise ValueError('Proposal or fit/validation cache differs from the recorded training run')
    if int(proposal['max_rank']) != cfg['max_rank'] or float(proposal['alpha']) != cfg['alpha']:
        raise ValueError('Selection config differs from the trained rank family')
    if len(arrays['source_bias']) != cfg['classes']:
        raise ValueError('Selection config class count differs from the cache')
    residuals = raw_residuals(proposal, arrays)
    budgets = (cfg['fit_nfr_budget'], cfg['validation_nfr_budget'])
    if not all(0 <= value <= 1 for value in budgets) or cfg['selection_tolerance'] < 0:
        raise ValueError('Budgets are fractions in [0,1]; rank tolerance is nonnegative pp')
    cells = []
    for rank, (dw, db) in sorted(residuals.items()):
        _, rank_cells = exact_path_selection(rank, dw, db, arrays, *budgets)
        cells.extend(rank_cells)
    chosen = select_rank_family(cells, *budgets, cfg['selection_tolerance'])
    rank, rho = int(chosen['rank']), float(chosen['radial_scale'])
    dw, db = (rho * value for value in residuals[rank])
    observed = {}
    for split, budget in zip(['fit', 'validation'], budgets):
        x, z0 = arrays[f'{split}_features'], arrays[f'source_{split}']
        observed[split] = class_metrics(arrays[f'{split}_labels'], z0, z0 + x @ dw.T + db)
        actual = observed[split]
        prefix = 'fit_' if split == 'fit' else ''
        expected_acc = chosen['fit_macro_accuracy' if split == 'fit' else 'validation_macro_accuracy']
        if (abs(actual['adapted_macro_accuracy'] - expected_acc) > 1e-8 or
                abs(actual['harmful_crossing_rate'] - chosen[prefix+'harmful_crossing_rate']) > 1e-8):
            raise RuntimeError('Event and direct predictions disagree (tie/root tolerance); no head exported')
        if actual['harmful_crossing_rate'] > 100 * budget + 1e-8:
            raise RuntimeError('Selected head violates an observed budget; no head exported')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir/'head.npz', source_weight=arrays['source_weight'],
        source_bias=arrays['source_bias'], delta_weight=dw, delta_bias=db,
        rank=np.asarray(rank), rho=np.asarray(rho), task=arrays['task'], stage=arrays['stage'])
    report = {'selected':chosen, 'observed':observed, 'config':cfg, 'cell_count':len(cells),
        'fit_validation_sha256':digest(cache_path), 'proposal_sha256':digest(proposal_path),
        'head_sha256':digest(output_dir/'head.npz'),
        'residual_semantics':'delta_weight and delta_bias already include rho; do not scale again'}
    write_json(output_dir/'selection.json', report)
    write_json(output_dir/'path_cells.json', cells)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'cache', 'proposal', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    report = select(a.proposal, a.cache, json.loads(a.config.read_text()), a.output_dir)
    print(json.dumps(report['selected'], indent=2))


if __name__ == '__main__':
    main()
