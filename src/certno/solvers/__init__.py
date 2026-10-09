"""Reference PDE solvers used to generate trusted training/reference data.

All solvers here are *verified* against manufactured solutions in the test
suite, so that a neural surrogate's error relative to these references is a
meaningful quantity.  Every solver returns enough information (the assembled
operator, grid, and stability constants) for the a posteriori certificate
machinery in :mod:`certno.certificates`.
"""

from .elliptic import (
    assemble_elliptic_1d,
    assemble_elliptic_2d,
    solve_elliptic_1d,
    solve_elliptic_2d,
    laplacian_min_eig_1d,
    laplacian_min_eig_2d,
    coercivity_bound_1d,
    coercivity_bound_2d,
)
from .spectral1d import solve_advection_diffusion_reaction_1d
from .manufactured import (
    manufactured_rhs_1d,
    manufactured_rhs_2d,
    elliptic_manufactured_1d,
)
from .fields import sample_grf_1d, sample_grf_2d, sample_grf_2d_flat

__all__ = [
    "assemble_elliptic_1d",
    "assemble_elliptic_2d",
    "solve_elliptic_1d",
    "solve_elliptic_2d",
    "laplacian_min_eig_1d",
    "laplacian_min_eig_2d",
    "coercivity_bound_1d",
    "coercivity_bound_2d",
    "solve_advection_diffusion_reaction_1d",
    "manufactured_rhs_1d",
    "manufactured_rhs_2d",
    "elliptic_manufactured_1d",
    "sample_grf_1d",
    "sample_grf_2d",
    "sample_grf_2d_flat",
]
