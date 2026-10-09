"""A compact 1D DeepONet: branch net over input functions, trunk net over space."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sizes: list[int], activation=nn.GELU):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet1d(nn.Module):
    """Operator network with a branch (function) net and a trunk (coordinate) net.

    Inputs are given as values on a fixed sensor grid.  The output at coordinate
    ``x`` is ``sum_k branch_k(input) * trunk_k(x) + bias``.
    """

    def __init__(
        self,
        n_sensors: int,
        p: int = 64,
        branch_width: int = 64,
        trunk_width: int = 64,
        branch_depth: int = 3,
        trunk_depth: int = 3,
        out_channels: int = 1,
        coord_dim: int = 1,
    ):
        super().__init__()
        branch_sizes = [n_sensors] + [branch_width] * (branch_depth - 1) + [p * out_channels]
        trunk_sizes = [coord_dim] + [trunk_width] * (trunk_depth - 1) + [p * out_channels]
        self.branch = MLP(branch_sizes)
        self.trunk = MLP(trunk_sizes)
        self.p = p
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, sensors: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # sensors: (B, n_sensors); coords: (N, coord_dim)
        b = self.branch(sensors).view(-1, self.p, self.out_channels)  # (B, p, C)
        t = self.trunk(coords).view(-1, self.p, self.out_channels)  # (N, p, C)
        out = torch.einsum("bpc,npc->bnc", b, t) + self.bias  # (B, N, C)
        return out
