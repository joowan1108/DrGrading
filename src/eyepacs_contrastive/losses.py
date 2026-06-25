from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss used by SimCLR."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}.")

        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)

        logits = torch.matmul(z, z.T) / self.temperature
        self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, float("-inf"))

        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)
        return F.cross_entropy(logits, targets)
