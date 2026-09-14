"""Evaluate all seen frozen branches with method-specific minimum-entropy routing."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.icassp2027.audio_data import ManifestWaveformDataset
from experiments.icassp2027.manifest import load_manifest
from experiments.icassp2027.dcase_boundary_model import BoundaryMigrationModel, load_source_backbone
from experiments.icassp2027.cache_stage_features import extract, normalization_overlay
from experiments.icassp2027.train_dcase_stage import choose_device
from .events import class_metrics
from .routing import sequential_metrics
from .io import digest, write_json


def evaluate_registry(registry, manifest, data_root, output_dir, device='auto', batch_size=16):
    config = json.loads(Path(registry).read_text())
    branches = config['branches']
    method_name = config.get('method_name', 'safbm')
    if not isinstance(method_name, str) or not method_name.isidentifier() or method_name == 'anchor':
        raise ValueError('method_name must be an identifier other than anchor')
    if [int(b['domain_index']) for b in branches] != list(range(len(branches))):
        raise ValueError('Branches must be ordered source=0, then 1..T-1 with no gaps')
    frame = load_manifest(Path(manifest))
    frame = frame[(frame.usage == 'test') & (frame.class_index >= 0)].copy()
    if frame.empty or frame.sample_id.duplicated().any():
        raise ValueError('Need nonempty, unique labeled test rows')
    if (frame.domain_index < 0).any() or (frame.domain_index >= len(branches)).any():
        raise ValueError('Test domains do not match the complete branch registry')
    source_path = Path(config['source_checkpoint'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device)
    collected = {'anchor':[], method_name:[]}
    known = []
    hashes = {'source_checkpoint':digest(source_path), 'manifest':digest(manifest), 'branches':[]}
    for task, branch in enumerate(branches):
        backbone = load_source_backbone(source_path)
        model = BoundaryMigrationModel(backbone, method='fixed', rank=1)
        branch_hash = {'domain_index':task}
        if task:
            bn_path, head_path = Path(branch['bn_checkpoint']), Path(branch['head'])
            model.load_state_dict(normalization_overlay(bn_path, task), strict=False)
            with np.load(head_path, allow_pickle=False) as head:
                if int(head['task']) != task:
                    raise ValueError('Head domain index differs from BN branch')
                w, b = backbone.fc.weight.detach().numpy(), backbone.fc.bias.detach().numpy()
                if not np.array_equal(head['source_weight'], w) or not np.array_equal(head['source_bias'], b):
                    raise ValueError('Deployed head and backbone source anchor differ')
                dw, db = head['delta_weight'].copy(), head['delta_bias'].copy()
            branch_hash.update(bn_checkpoint=digest(bn_path), head=digest(head_path))
        else:
            dw = np.zeros_like(backbone.fc.weight.detach().numpy(), dtype=np.float64)
            db = np.zeros_like(backbone.fc.bias.detach().numpy(), dtype=np.float64)
        loader = DataLoader(ManifestWaveformDataset(frame, Path(data_root)), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        model.to(device).eval()
        x, labels, sample_ids = extract(model, loader, task, device)
        x = x.astype(np.float64)
        w = backbone.fc.weight.detach().cpu().numpy().astype(np.float64)
        b = backbone.fc.bias.detach().cpu().numpy().astype(np.float64)
        z0 = x @ w.T + b
        z = z0 + x @ dw.T + db
        collected['anchor'].append(z0)
        collected[method_name].append(z)
        mask = frame.domain_index.to_numpy() == task
        if mask.any():
            known.append({'domain_index':task, **class_metrics(labels[mask], z0[mask], z[mask])})
        hashes['branches'].append(branch_hash)
        del model, backbone
        print(f'Extracted branch {task}: {len(labels)} held-out examples', flush=True)
    domains = frame.domain_index.to_numpy(dtype=np.int64)
    result = {name:sequential_metrics(np.stack(values), labels, domains)
              for name, values in collected.items()}
    result.update(known_domain=known, provenance=hashes,
                  protocol='Frozen branches; method-specific entropy routing, temperature 1; test labels evaluation only')
    np.savez_compressed(output_dir/'branch_logits.npz', labels=labels, domains=domains,
        sample_ids=sample_ids, **{name+'_logits':np.stack(values) for name,values in collected.items()})
    write_json(output_dir/'metrics.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['registry','manifest','data-root','output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    evaluate_registry(a.registry, a.manifest, a.data_root, a.output_dir, a.device, a.batch_size)


if __name__ == '__main__':
    main()
