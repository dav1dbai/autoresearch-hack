"""Reference task: square matrix multiplication (KernelBench L1 style).

The agent must produce a drop-in `ModelNew` that is numerically equivalent to
`Model.forward` and faster. Inputs/shapes are fixed here for a fast rollout.
"""

import torch
import torch.nn as nn

N = 1024


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


def get_inputs():
    return [torch.randn(N, N), torch.randn(N, N)]


def get_init_inputs():
    return []
