"""A compact 1D Fourier Neural Operator (FNO) for elliptic solution maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Spectral convolution that multiplies a truncated Fourier basis by weights."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N)
        batch = x.shape[0]
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :m] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """Fourier neural operator mapping input channels on a grid to an output field.

    Parameters
    ----------
    in_channels:
        Number of input fields (e.g. coefficient ``a`` and forcing ``f``).
    out_channels:
        Number of output fields (the solution).
    width, modes, depth:
        Hidden width, retained Fourier modes, and number of spectral blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 16,
        depth: int = 4,
        padding: int = 0,
    ):
        super().__init__()
        self.padding = padding
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv1d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = x.permute(0, 2, 1)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.project(x)
        if self.padding > 0:
            x = x[..., : -self.padding]
        return x.permute(0, 2, 1)
