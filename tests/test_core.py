import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from safbm.events import enumerate_correctness_cells, class_metrics, select_rank_family
from safbm.io import load_fit_validation
from safbm.routing import sequential_metrics
from safbm.select import raw_residuals, select
from safbm.model import SAFBMHead
from experiments.icassp2027.train_cached_hcrg import RankGrowingBoundaryResidual


class CoreTests(unittest.TestCase):
    def test_event_cells_match_direct_argmax(self):
        rng = np.random.default_rng(24)
        for _ in range(8):
            start, end = rng.normal(size=(2, 60, 4))
            labels = rng.integers(0, 4, 60)
            cells = enumerate_correctness_cells(2, start, end, labels, start, end, labels)
            for cell in cells:
                for rho in [cell['radial_scale'], (cell['left']+cell['right'])/2]:
                    actual = class_metrics(labels, start, start + rho*(end-start))
                    self.assertAlmostEqual(actual['adapted_macro_accuracy'], cell['validation_macro_accuracy'])
                    self.assertAlmostEqual(actual['harmful_crossing_rate'], cell['harmful_crossing_rate'])

    def test_macro_identity_and_nfr_denominator(self):
        y = np.array([0, 0, 0, 1])
        z0 = np.eye(2)[[0, 0, 1, 0]]
        z = np.eye(2)[[1, 0, 0, 1]]
        m = class_metrics(y, z0, z)
        self.assertAlmostEqual(m['harmful_crossing_rate'], 100/6)
        self.assertAlmostEqual(m['accuracy_gain'], m['correcting_crossing_rate']-m['harmful_crossing_rate'])

    def test_joint_selection_can_use_non_endpoint_rank(self):
        def cell(rank, rho, acc, nfr=0):
            return dict(rank=rank, radial_scale=rho, validation_macro_accuracy=acc,
                        net_correcting_rate=acc-60, harmful_crossing_rate=nfr,
                        fit_harmful_crossing_rate=nfr)
        cells = [cell(0,0,60), cell(1,1,65), cell(1,.4,72), cell(2,1,70), cell(2,.6,71)]
        selected = select_rank_family(cells, .005, .010, .5)
        self.assertEqual((selected['rank'], selected['radial_scale']), (1,.4))
        # The better endpoint belongs to rank 2; endpoint-first would omit rank 1.
        self.assertEqual(max((c for c in cells if c['radial_scale']==1), key=lambda c:c['validation_macro_accuracy'])['rank'], 2)

    def test_raw_endpoint_and_anchor_validation(self):
        w, b = torch.zeros(3,4), torch.zeros(3)
        m = RankGrowingBoundaryResidual(w, b, 1, 1)
        m.active_rank=1
        with torch.no_grad():
            m.A.fill_(1)
            m.B[:,0].copy_(torch.tensor([1.,-1.,0.]))
        state = {k:v.clone() for k,v in m.state_dict().items()}
        restored = {k:v.clone() for k,v in state.items()}
        restored['B'].zero_()
        p = dict(max_rank=1, alpha=1, factor_scaling=1, source_weight=w,source_bias=b,
                 candidate_raw_states={1:state},candidate_restored_states={1:restored})
        arrays = dict(source_weight=w.numpy(),source_bias=b.numpy())
        self.assertGreater(np.linalg.norm(raw_residuals(p,arrays)[1][0]),0)
        state['source_weight'][0,0]=1
        with self.assertRaises(ValueError):
            raw_residuals(p,arrays)

    def test_frozen_branches_can_forget_through_routing(self):
        # Branch 1's confident wrong prediction reroutes a previously correct D0 input.
        logits = np.array([[[2.,0.],[0.,2.]], [[0.,10.],[0.,10.]]])
        r = sequential_metrics(logits, np.array([0,1]), np.array([0,1]))
        self.assertEqual(r['accuracy_matrix'], [[100.,None],[0.,100.]])
        self.assertEqual(r['final_avg'],50.)
        self.assertEqual(r['forgetting'],100.)
        self.assertEqual(r['backward_transfer'],-100.)
        self.assertEqual(r['route_counts_by_arrival'][0], [1,0])

    def test_missing_source_domain_is_not_zero_accuracy(self):
        logits = np.array([[[2.,0.]],[[3.,0.]]])
        r = sequential_metrics(logits,np.array([0]),np.array([1]))
        self.assertIsNone(r['accuracy_matrix'][1][0])
        self.assertEqual(r['final_avg'],100.)
        self.assertIsNone(r['forgetting'])

    def test_nonzero_selected_head_matches_frozen_path(self):
        w = torch.eye(3).roll(1, dims=0)
        b = torch.zeros(3)
        dw = torch.eye(3) - w
        u, sv, vh = torch.linalg.svd(dw)
        model = RankGrowingBoundaryResidual(w, b, 2, 2)
        state1 = {k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            model.A.copy_(vh[:2])
            model.B.copy_(u[:,:2]*sv[:2])
        state2 = {k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            x, y = np.eye(3, dtype=np.float32), np.arange(3)
            np.savez(p/'cache.npz',fit_features=x,fit_labels=y,validation_features=x,
                     validation_labels=y,source_weight=w.numpy(),source_bias=b.numpy(),task=1,stage=1)
            torch.save(dict(max_rank=2,alpha=2,factor_scaling=1,source_weight=w,source_bias=b,
                            candidate_raw_states={1:state1,2:state2},task=1,stage=1), p/'proposal.pth')
            cfg = dict(classes=3,max_rank=2,alpha=2,fit_nfr_budget=0,validation_nfr_budget=0,selection_tolerance=.5)
            result = select(p/'proposal.pth',p/'cache.npz',cfg,p/'selected')
            self.assertEqual(result['selected']['rank'],2)
            self.assertEqual(result['observed']['validation']['adapted_macro_accuracy'],100)
            self.assertEqual(result['observed']['validation']['harmful_crossing_rate'],0)
            deployed = SAFBMHead.from_npz(p/'selected/head.npz')
            self.assertEqual(deployed(torch.eye(3,dtype=torch.float64)).argmax(-1).tolist(), [0,1,2])
            with np.load(p/'selected/head.npz') as h:
                model.active_rank = 2
                expected = result['selected']['radial_scale'] * model.delta_weight().detach().numpy()
                np.testing.assert_allclose(h['delta_weight'],expected,atol=1e-7)

    def test_selector_input_rejects_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'cache.npz'
            np.savez(path,test_features=np.zeros((2,3)))
            with self.assertRaisesRegex(ValueError,'held-out'):
                load_fit_validation(path)


if __name__ == '__main__':
    unittest.main()
