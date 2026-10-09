"""Training / inference helpers for the learned solution operators."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .datasets import EllipticDataset1D


def _to_tensors(dataset: EllipticDataset1D, kind: str):
    inputs = torch.tensor(dataset.inputs(), dtype=torch.float32)  # (n, N, 2)
    targets = torch.tensor(dataset.u[..., None], dtype=torch.float32)  # (n, N, 1)
    return inputs, targets


def _forward(model, inputs: torch.Tensor, coords: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "fno":
        return model(inputs)
    if kind == "deeponet":
        sensors = inputs.reshape(inputs.shape[0], -1)  # (B, N*2)
        return model(sensors, coords)
    raise ValueError(f"unknown operator kind: {kind!r}")


def train_operator(
    model: nn.Module,
    dataset: EllipticDataset1D,
    *,
    kind: str = "fno",
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
    mask_boundary: bool = True,
    verbose: bool = False,
) -> dict:
    """Train ``model`` on ``dataset`` and return a history dictionary."""
    torch.manual_seed(seed)
    inputs, targets = _to_tensors(dataset, kind)
    n = inputs.shape[0]
    coords = torch.tensor(dataset.x[:, None], dtype=torch.float32)
    mask = torch.ones_like(targets)
    if mask_boundary:
        mask[:, 0, :] = 0.0
        mask[:, -1, :] = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb, mb = inputs[idx], targets[idx], mask[idx]
            optimizer.zero_grad()
            pred = _forward(model, xb, coords, kind)
            loss = (loss_fn(pred * mb, yb * mb))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * idx.numel()
        history["loss"].append(epoch_loss / n)
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f"epoch {epoch:4d}  loss {history['loss'][-1]:.6e}")
    history["final_loss"] = history["loss"][-1]
    return history


@torch.no_grad()
def predict_operator(
    model: nn.Module,
    dataset: EllipticDataset1D,
    *,
    kind: str = "fno",
    mask_boundary: bool = True,
) -> np.ndarray:
    """Return predictions ``(n, N)`` as a numpy array."""
    model.eval()
    inputs, _ = _to_tensors(dataset, kind)
    coords = torch.tensor(dataset.x[:, None], dtype=torch.float32)
    pred = _forward(model, inputs, coords, kind).squeeze(-1).cpu().numpy()
    if mask_boundary:
        pred[:, 0] = 0.0
        pred[:, -1] = 0.0
    return pred
