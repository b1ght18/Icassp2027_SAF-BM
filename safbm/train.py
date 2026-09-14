"""Run the original rank-growing trajectory with the paper's explicit defaults."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .io import load_fit_validation, digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs-per-rank', type=int)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    arrays = load_fit_validation(args.cache)
    if len(arrays['source_bias']) != cfg['classes']:
        p.error('config.classes does not match the supplied source classifier')
    if args.epochs_per_rank is not None:
        cfg['epochs_per_rank'] = args.epochs_per_rank
    if args.seed is not None:
        cfg['seed'] = args.seed
    if not 1 <= cfg['max_rank'] <= min(9, cfg['classes'] - 1, arrays['fit_features'].shape[1]):
        p.error('max_rank must be in 1..min(9, C-1)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'proposal.pth').exists():
        p.error('output already contains proposal.pth; choose a new output directory')
    keys = ['constraint_mode','max_rank','alpha','learning_rate','dual_learning_rate',
            'dual_max','harm_slack','growth_magnitude','epochs_per_rank','patience',
            'batch_size','selection_tolerance','feasibility_tolerance',
            'feasibility_restoration','radial_grid_size','seed']
    command = [sys.executable, '-m', 'experiments.icassp2027.train_cached_hcrg',
               '--cache', str(args.cache), '--output', str(args.output_dir/'proposal.pth'),
               '--summary', str(args.output_dir/'training.csv')]
    for key in keys:
        command += ['--'+key.replace('_','-'), str(cfg[key])]
    subprocess.run(command, check=True)
    write_json(args.output_dir/'run.json', {'config':cfg, 'fit_validation_sha256':digest(args.cache),
               'proposal_sha256':digest(args.output_dir/'proposal.pth')})


if __name__ == '__main__':
    main()
