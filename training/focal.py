"""Focal, class-weighted BCE for the YOLO classification loss (needs torch).

Kept in its own module so it is importable by name: Ultralytics pickles the
model, with its loss, into every checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WeightedFocalBCE(nn.Module):
    """Element-wise BCE-with-logits (``reduction='none'``), focal and class-weighted."""

    def __init__(self, gamma: float = 0.0, alpha: float | None = None, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, true, reduction="none")
        if self.gamma > 0:
            p = pred.sigmoid()
            p_t = true * p + (1 - true) * (1 - p)
            loss = loss * (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            loss = loss * (true * self.alpha + (1 - true) * (1 - self.alpha))
        if self.weights is not None and self.weights.numel() == loss.shape[-1]:
            loss = loss * self.weights.to(loss.device, loss.dtype)
        return loss
