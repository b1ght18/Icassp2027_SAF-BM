"""Export a validation-selected baseline head for held-out evaluation; no reselection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.icassp2027.boundary_residual import FixedRankBoundaryResidual
from .io import digest, write_json


def export_control(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    p = torch.load(checkpoint, map_location='cpu', weights_only=False)
    formats = {'icassp2027-cached-boundary-control-v1',
               'icassp2027-validation-adaptive-boundary-rank-v1',
               'icassp2027-posthoc-svd-boundary-rank-v1'}
    if p.get('format') not in formats:
        raise ValueError('Not a supported selected control; SAF-BM proposals require safbm.select')
    state = p['selected_state_dict']
    rank = int(p['selected_rank'])
    m = FixedRankBoundaryResidual(state['source_weight'],state['source_bias'],rank,p['alpha'])
    m.load_state_dict(state,strict=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,source_weight=m.source_weight.double().numpy(),
        source_bias=m.source_bias.double().numpy(),delta_weight=m.delta_weight().detach().double().numpy(),
        delta_bias=m.delta_bias.detach().double().numpy(),rank=np.asarray(rank),rho=np.asarray(1.),
        task=np.asarray(p['task']),stage=np.asarray(p['stage']))
    write_json(output.with_suffix('.json'), {'input_sha256':digest(checkpoint),'head_sha256':digest(output),
        'method':p.get('control_method',p['format']), 'selection':'pre-existing validation-selected endpoint; no NFR reselection'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    export_control(a.checkpoint,a.output)


if __name__ == '__main__':
    main()
