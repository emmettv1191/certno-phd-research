"""Dataset generation for the 1D elliptic solution operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..solvers.elliptic import solve_elliptic_1d
from ..solvers.fields import sample_coefficient_1d


@dataclass
class EllipticDataset1D:
    """Arrays for the map ``(a, f) -> u`` on the full grid of ``N+2`` nodes."""

    a: np.ndarray  # (n, N+2) coefficient, boundaries included
    f: np.ndarray  # (n, N+2) forcing, zero at boundaries
    u: np.ndarray  # (n, N+2) solution, zero at boundaries
    x: np.ndarray  # (N+2,)
    n_interior: int
    h: float
    meta: dict

    @property
    def n_samples(self) -> int:
        return self.a.shape[0]

    def inputs(self) -> np.ndarray:
        """Stack input channels as ``(n, N+2, 2)``."""
        return np.stack([self.a, self.f], axis=-1)

    def subset(self, idx: np.ndarray) -> "EllipticDataset1D":
        return EllipticDataset1D(
            a=self.a[idx],
            f=self.f[idx],
            u=self.u[idx],
            x=self.x,
            n_interior=self.n_interior,
            h=self.h,
            meta=dict(self.meta),
        )


def make_forcing(
    x: np.ndarray, kind: str, rng: np.random.Generator | None = None, n_modes: int = 5
) -> np.ndarray:
    """Return a forcing profile on ``x``, zero at the endpoints."""
    rng = rng or np.random.default_rng(0)
    if kind == "sin":
        f = np.sin(np.pi * x)
    elif kind == "random_sines":
        f = np.zeros_like(x)
        for k in range(1, n_modes + 1):
            f += rng.standard_normal() * np.sin(k * np.pi * x) / k
    else:
        raise ValueError(f"unknown forcing kind: {kind!r}")
    f[0] = 0.0
    f[-1] = 0.0
    return f


def generate_elliptic_1d_dataset(
    n_samples: int,
    n_interior: int = 63,
    *,
    length_scale: float = 0.2,
    a_min: float = 0.1,
    a_max: float = 1.0,
    smoothness: float = 1.5,
    forcing: str = "sin",
    n_forcing_modes: int = 5,
    seed: int = 0,
) -> EllipticDataset1D:
    """Sample coefficients, solve the PDE, and return aligned arrays."""
    rng = np.random.default_rng(seed)
    n_nodes = n_interior + 2
    x = np.linspace(0.0, 1.0, n_nodes)

    A = np.empty((n_samples, n_nodes))
    F = np.empty((n_samples, n_nodes))
    U = np.empty((n_samples, n_nodes))

    for s in range(n_samples):
        a_nodes = sample_coefficient_1d(
            n_nodes,
            length_scale=length_scale,
            a_min=a_min,
            a_max=a_max,
            smoothness=smoothness,
            rng=rng,
        )
        f = make_forcing(x, forcing, rng=rng, n_modes=n_forcing_modes)
        sol = solve_elliptic_1d(a_nodes, f[1:-1], n_interior)
        A[s] = a_nodes
        F[s] = f
        U[s] = sol.u

    meta = {
        "n_interior": n_interior,
        "length_scale": length_scale,
        "a_min": a_min,
        "a_max": a_max,
        "smoothness": smoothness,
        "forcing": forcing,
        "n_forcing_modes": n_forcing_modes,
        "seed": seed,
    }
    return EllipticDataset1D(
        a=A, f=F, u=U, x=x, n_interior=n_interior, h=1.0 / (n_interior + 1), meta=meta
    )
