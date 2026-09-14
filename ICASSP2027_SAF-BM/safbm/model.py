"""A frozen deployed head; the residual is merged into the source linear head."""
import numpy as np
import torch
from torch import nn


class SAFBMHead(nn.Linear):
    @classmethod
    def from_npz(cls, path, dtype=torch.float64):
        with np.load(path, allow_pickle=False) as a:
            w, b = a['source_weight'] + a['delta_weight'], a['source_bias'] + a['delta_bias']
        head = cls(w.shape[1], w.shape[0]).to(dtype=dtype)
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(w, dtype=dtype))
            head.bias.copy_(torch.as_tensor(b, dtype=dtype))
        return head.requires_grad_(False).eval()
