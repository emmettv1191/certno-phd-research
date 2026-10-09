"""certno: Error-certified neural operators for reliable computational engineering.

The package is organized around the research question:

    How can we build computational engineering models that learn from data,
    obey physical laws, quantify their own errors, and remain trustworthy
    when conditions differ from those seen during training?

Modules
-------
solvers      : high-accuracy reference PDE solvers (elliptic, Burgers, reaction-diffusion)
operators    : learned surrogate models (FNO, DeepONet) and dataset generation
certificates : a posteriori residual/stability error certificates + adaptive verification
active       : goal-oriented active learning (Problem B)
diagnostics  : failure-mode identification (Problem D)
metrics      : error / coverage / cost measurement
"""

from ._version import __version__

__all__ = ["__version__"]
