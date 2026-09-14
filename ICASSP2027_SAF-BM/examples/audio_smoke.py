"""Optional waveform-to-routing smoke check using temporary synthetic audio and random weights."""
from pathlib import Path
import tempfile, json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from safbm.backbone import MCnn14
from safbm.evaluate_sequence import evaluate_registry
from safbm.export_control import export_control
from safbm.model import SAFBMHead
from experiments.icassp2027.boundary_residual import FixedRankBoundaryResidual

def main():
    with tempfile.TemporaryDirectory(prefix='safbm-audio-smoke-') as d:
        p = Path(d)
        torch.manual_seed(4)
        b = MCnn14(classes_num=3, nb_tasks=3).eval()
        state = b.state_dict()
        torch.save(state, p / 'source.pth')
        rows = []
        branches = [{'domain_index': 0}]
        rng = np.random.default_rng(5)
        for domain in range(3):
            name = f'synthetic_{domain}.wav'
            sf.write(p / name, rng.normal(0, 0.02, 32000).astype(np.float32), 32000)
            rows.append(dict(sample_id=str(domain), relative_path=name, label=str(domain), class_index=domain, domain=f'D{domain + 1}', domain_index=domain, stage=domain, official_partition='test', usage='test', recording_group=str(domain), content_group=str(domain), dataset='TAU-ASC2022-Mobile'))
            if domain:
                bn = {k: v for k, v in state.items() if any((t in k for t in (f'bn0.{domain}.', f'bnF.{domain}.', f'bnS.{domain}.')))}
                torch.save(bn, p / f'bn{domain}.pth')
                head = FixedRankBoundaryResidual(b.fc.weight.detach(), b.fc.bias.detach(), 1, 8)
                with torch.no_grad():
                    head.B.normal_(0, 0.001)
                    head.delta_bias[domain] = 0.1
                torch.save(dict(format='icassp2027-cached-boundary-control-v1', selected_rank=1, selected_state_dict=head.state_dict(), alpha=8, task=domain, stage=domain), p / f'control{domain}.pth')
                export_control(p / f'control{domain}.pth', p / f'head{domain}.npz')
                exported = SAFBMHead.from_npz(p / f'head{domain}.npz')
                x = torch.randn(4, 2048)
                torch.testing.assert_close(head(x).double(), exported(x.double()), rtol=1e-05, atol=1e-05)
                branches.append(dict(domain_index=domain, bn_checkpoint=str(p / f'bn{domain}.pth'), head=str(p / f'head{domain}.npz')))
        pd.DataFrame(rows).to_csv(p / 'manifest.tsv', sep='\t', index=False)
        (p / 'registry.json').write_text(json.dumps(dict(source_checkpoint=str(p / 'source.pth'), branches=branches, method_name='control')))
        del state, b
        result = evaluate_registry(p / 'registry.json', p / 'manifest.tsv', p, p / 'eval', device='cpu', batch_size=2)
        assert len(result['control']['accuracy_matrix']) == 3
        assert len(result['known_domain']) == 3
        assert result['control']['accuracy_matrix'][0][1] is None
        print('PASS: waveform -> per-domain BN/CNN14 -> merged heads -> method-specific routing and matrix')
if __name__ == '__main__':
    main()
