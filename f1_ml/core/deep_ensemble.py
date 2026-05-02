"""Deep ensemble — train K independent neural nets with different inits;
ensemble-mean is the prediction, ensemble-variance is the uncertainty.

Cheaper than full Bayesian deep learning, comparable calibration quality. The
practical default for getting calibrated uncertainty out of a pace / overtake
neural net.

K = 5 is the sweet spot; more gives diminishing returns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


class DeepEnsemble:
    """Train K independently-initialized models on the same data.

    Each member gets a different random seed; the ensemble's predictive mean
    and standard deviation give point + uncertainty estimates.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        k: int = 5,
        loss: str = "mse",
        optimizer: str = "adam",
        lr: float = 1e-3,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if loss not in ("mse", "bce"):
            raise ValueError("loss must be 'mse' or 'bce'")
        self.model_factory = model_factory
        self.k = int(k)
        self.loss_kind = loss
        self.optimizer_kind = optimizer
        self.lr = float(lr)
        self.models: list[nn.Module] = []

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 50,
        batch_size: int | None = None,
        base_seed: int = 0,
    ) -> "DeepEnsemble":
        """Train k models with independent random inits."""
        self.models = []
        if self.loss_kind == "mse":
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        for k in range(self.k):
            torch.manual_seed(base_seed + k)
            model = self.model_factory()
            opt = self._make_optimizer(model)
            for _ in range(epochs):
                if batch_size is None:
                    pred = model(X).squeeze(-1) if model(X).dim() > y.dim() else model(X)
                    loss = loss_fn(pred, y)
                    opt.zero_grad(); loss.backward(); opt.step()
                else:
                    perm = torch.randperm(X.shape[0])
                    for start in range(0, X.shape[0], batch_size):
                        idx = perm[start:start + batch_size]
                        pred = model(X[idx]).squeeze(-1) if model(X[idx]).dim() > y[idx].dim() else model(X[idx])
                        loss = loss_fn(pred, y[idx])
                        opt.zero_grad(); loss.backward(); opt.step()
            self.models.append(model)
        return self

    def predict(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) over the ensemble's predictive distribution."""
        if not self.models:
            raise RuntimeError("DeepEnsemble is not fitted; call .fit(...) first")
        with torch.no_grad():
            preds = []
            for m in self.models:
                m.eval()
                p = m(X)
                if self.loss_kind == "bce":
                    p = torch.sigmoid(p)
                preds.append(p.detach().cpu().numpy())
            stack = np.stack(preds, axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        return np.asarray(mean), np.asarray(std)

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.optimizer_kind == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.optimizer_kind == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.lr)
        raise ValueError(f"unknown optimizer '{self.optimizer_kind}'")
