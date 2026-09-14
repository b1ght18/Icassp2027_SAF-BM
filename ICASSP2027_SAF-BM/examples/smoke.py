"""CPU end-to-end example with temporary synthetic features; no audio/checkpoints needed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from safbm.select import select
from safbm.model import SAFBMHead
import torch


def main():
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(19)
    with tempfile.TemporaryDirectory(prefix='safbm-smoke-') as directory:
        work = Path(directory)
        w = rng.normal(size=(3, 8)).astype(np.float32)
        b = np.zeros(3, dtype=np.float32)
        teacher = w + .7 * rng.normal(size=w.shape)
        def split(n):
            x = rng.normal(size=(n, 8)).astype(np.float32)
            return x, (x @ teacher.T).argmax(1)
        xf, yf = split(96)
        xv, yv = split(48)
        cache = work/'fit_validation.npz'
        np.savez(cache, fit_features=xf, fit_labels=yf, validation_features=xv,
                 validation_labels=yv, source_weight=w, source_bias=b, task=1, stage=1)
        cfg = json.loads((root/'configs/adil.json').read_text())
        cfg.update(classes=3, max_rank=2, alpha=16/3, epochs_per_rank=2, patience=2,
                   batch_size=32, radial_grid_size=11)
        (work/'config.json').write_text(json.dumps(cfg))
        subprocess.run([sys.executable, '-m', 'safbm.train', '--config', str(work/'config.json'),
                        '--cache', str(cache), '--output-dir', str(work/'train')], check=True)
        report = select(work/'train/proposal.pth', cache, cfg, work/'selected')
        head = SAFBMHead.from_npz(work/'selected/head.npz')
        with np.load(work/'selected/head.npz') as h:
            expected = xf.astype(float) @ (h['source_weight']+h['delta_weight']).T + h['source_bias']+h['delta_bias']
        np.testing.assert_allclose(head(torch.tensor(xf, dtype=torch.float64)).numpy(), expected, atol=1e-10)
        xt, yt = split(24)
        np.savez(work/'heldout.npz', features=xt, labels=yt, task=1)
        subprocess.run([sys.executable, '-m', 'safbm.evaluate', '--head', str(work/'selected/head.npz'),
                        '--features', str(work/'heldout.npz'), '--output', str(work/'evaluation.json')], check=True)
        print('PASS: training -> frozen raw paths -> joint selection -> merged head -> held-out evaluation')
        print('Synthetic smoke check only; these values are not experimental evidence.')


if __name__ == '__main__':
    main()
