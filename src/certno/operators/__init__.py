"""Learned solution operators (surrogates) and their datasets.

``torch`` is imported lazily so that the solver/certificate parts of the package
remain usable without a deep-learning stack."""

from __future__ import annotations

from .datasets import generate_elliptic_1d_dataset, EllipticDataset1D

__all__ = ["generate_elliptic_1d_dataset", "EllipticDataset1D"]


def __getattr__(name: str):
    # Lazy re-exports that require torch.
    if name in {"FNO1d", "DeepONet1d", "train_operator", "predict_operator"}:
        from . import fno, deeponet, train as _train

        return {
            "FNO1d": getattr(fno, "FNO1d"),
            "DeepONet1d": getattr(deeponet, "DeepONet1d"),
            "train_operator": getattr(_train, "train_operator"),
            "predict_operator": getattr(_train, "predict_operator"),
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
