"""Opt-in Balanced Softmax model ported from gem_no_spatial_0814_bs_ablation.

Source: 03fd7e064acbfb1e0062d009e6e413ccc58a4a41. Keep the original regression
prediction; the five-class head supplies an auxiliary training loss only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import HybridOrdinalNet


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class BalancedSoftmaxHybridOrdinalNet(HybridOrdinalNet):
    """Same architecture/state-dict keys as the teammate's final model."""
    def __init__(self, *, num_classes: int = 5, **kwargs):
        super().__init__(**kwargs)
        head_dim = 2 * int(kwargs.get("projection_dim", 128)) if self.regression_input == "projection_concat" else self.feature_dim
        self.classification_head = ClassificationHead(head_dim, num_classes)

    def forward(self, images):
        outputs = super().forward(images)
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            inputs = (
                torch.cat((outputs["pcol"], outputs["scol"]), dim=1)
                if self.regression_input == "projection_concat" else outputs["features"]
            )
            outputs["logits"] = self.classification_head(inputs.float())
        return outputs


class BalancedSoftmaxLoss(nn.Module):
    """CE(logits + log(training-class counts), target), as in the source branch."""
    def __init__(self, class_counts):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or len(counts) < 2 or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("class_counts must be a finite, nonnegative one-dimensional vector.")
        self.register_buffer("log_class_counts", counts.clamp_min(1.0).log())

    def forward(self, logits, targets):
        adjusted = logits + self.log_class_counts.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(adjusted, targets)
