"""Portable, explicit inputs for training/selection and deployment."""
from pathlib import Path
import hashlib
import json
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_fit_validation(path):
    required = ['fit_features', 'fit_labels', 'validation_features',
                'validation_labels', 'source_weight', 'source_bias', 'task', 'stage']
    with np.load(path, allow_pickle=False) as archive:
        forbidden = [name for name in archive.files if any(t in name.lower() for t in ['test', 'heldout', 'held_out'])]
        if forbidden:
            raise ValueError(f'Fit/validation archive contains held-out arrays: {forbidden}')
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f'Missing arrays: {sorted(missing)}')
        a = {key: archive[key] for key in required}
    for split in ['fit', 'validation']:
        x = np.asarray(a[f'{split}_features'], dtype=np.float64)
        raw_y = np.asarray(a[f'{split}_labels'])
        if not np.issubdtype(raw_y.dtype, np.integer):
            raise ValueError(f'{split} labels must have an integer dtype')
        y = raw_y.astype(np.int64)
        if x.ndim != 2 or y.ndim != 1 or not len(y) or len(x) != len(y):
            raise ValueError(f'Invalid {split} shapes')
        if not np.isfinite(x).all() or y.min() < 0 or y.max() >= len(a['source_bias']):
            raise ValueError(f'Invalid {split} values')
        a[f'{split}_features'], a[f'{split}_labels'] = x, y
    a['source_weight'] = np.asarray(a['source_weight'], dtype=np.float64)
    a['source_bias'] = np.asarray(a['source_bias'], dtype=np.float64)
    if a['source_weight'].shape != (len(a['source_bias']), a['fit_features'].shape[1]):
        raise ValueError('Source head and feature dimensions disagree')
    if a['source_bias'].ndim != 1 or not np.isfinite(a['source_weight']).all() or not np.isfinite(a['source_bias']).all():
        raise ValueError('Invalid source parameters')
    if a['validation_features'].shape[1] != a['fit_features'].shape[1]:
        raise ValueError('Feature dimensions differ between splits')
    for key in ['task', 'stage']:
        if np.ndim(a[key]) != 0 or not np.issubdtype(a[key].dtype, np.integer) or int(a[key]) < 0:
            raise ValueError(f'{key} must be a nonnegative integer scalar')
    for split in ['fit', 'validation']:
        a[f'source_{split}'] = a[f'{split}_features'] @ a['source_weight'].T + a['source_bias']
    return a
