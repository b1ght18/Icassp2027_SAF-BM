"""Evaluate one already selected head on known-domain held-out features."""
import argparse
from pathlib import Path
import numpy as np
from .events import class_metrics
from .io import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True, help='Held-out NPZ: features, labels, task')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with np.load(a.head, allow_pickle=False) as h, np.load(a.features, allow_pickle=False) as d:
        if int(h['task']) != int(d['task']):
            raise ValueError('Held-out features and head use different BN branches')
        x, y = d['features'].astype(np.float64), d['labels'].astype(np.int64)
        z0 = x @ h['source_weight'].T + h['source_bias']
        z = z0 + x @ h['delta_weight'].T + h['delta_bias']
        if not len(y) or x.shape[0] != len(y) or not np.isfinite(z).all():
            raise ValueError('Invalid held-out arrays')
        report = class_metrics(y, z0, z)
    report.update(head_sha256=digest(a.head), heldout_sha256=digest(a.features))
    write_json(a.output, report)
    print(report)


if __name__ == '__main__':
    main()
