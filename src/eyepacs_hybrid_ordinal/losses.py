from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CumulativeOrdinalMargins(nn.Module):
    """Learn positive adjacent-class margins and accumulate them across ranks."""

    def __init__(
        self,
        num_classes: int,
        minimum_margin: float = 0.05,
        init_min: float = 0.5,
        init_max: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if minimum_margin < 0:
            raise ValueError("minimum_margin must be non-negative.")
        if not minimum_margin < init_min <= init_max:
            raise ValueError(
                "Expected minimum_margin < init_min <= init_max for learnable margins."
            )

        self.num_classes = int(num_classes)
        self.minimum_margin = float(minimum_margin)
        initial_margins = torch.empty(self.num_classes - 1).uniform_(init_min, init_max)
        shifted = initial_margins - self.minimum_margin
        self.raw_margins = nn.Parameter(torch.log(torch.expm1(shifted)))

    def margin_values(self) -> torch.Tensor:
        return self.minimum_margin + F.softplus(self.raw_margins)

    def class_positions(self) -> torch.Tensor:
        zero = self.raw_margins.new_zeros(1)
        return torch.cat((zero, torch.cumsum(self.margin_values(), dim=0)))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.long()
        right = right.long()
        positions = self.class_positions()
        return torch.abs(
            positions[left].unsqueeze(-1) - positions[right].unsqueeze(0)
        )

    def freeze(self) -> None:
        self.raw_margins.requires_grad_(False)


def ordinal_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    num_classes: int,
    normalize: bool = False,
    margin_scale: float = 1.0,
) -> torch.Tensor:
    distance = torch.abs(left.float().unsqueeze(-1) - right.float().unsqueeze(0))
    if normalize and num_classes > 1:
        distance = distance / float(num_classes - 1)
    return distance * float(margin_scale)


class PrototypeContrastiveOrdinalLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        num_classes: int = 5,
        margin_scale: float = 1.0,
        normalize_ordinal_distance: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.margin_scale = float(margin_scale)
        self.normalize_ordinal_distance = bool(normalize_ordinal_distance)
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        learnable_margins: CumulativeOrdinalMargins | None = None,
    ) -> torch.Tensor:
        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.long()
        prototype_labels = torch.unique(labels, sorted=True)

        if prototype_labels.numel() < 2:
            return embeddings.sum() * 0.0

        prototypes = []
        for label in prototype_labels:
            prototype = embeddings[labels == label].mean(dim=0)
            prototypes.append(F.normalize(prototype, dim=0))
        prototypes_tensor = torch.stack(prototypes, dim=0)

        similarities = embeddings @ prototypes_tensor.T
        positive_mask = labels.unsqueeze(1) == prototype_labels.unsqueeze(0)
        negative_mask = labels.unsqueeze(1) != prototype_labels.unsqueeze(0)
        if learnable_margins is None:
            distances = ordinal_distance(
                labels,
                prototype_labels,
                num_classes=self.num_classes,
                normalize=self.normalize_ordinal_distance,
                margin_scale=self.margin_scale,
            )
        else:
            distances = learnable_margins(labels, prototype_labels) * self.margin_scale

        positive_logits = similarities[positive_mask] / self.temperature
        negative_logits = (similarities + distances)[negative_mask].view(embeddings.shape[0], -1)
        negative_log_denominator = torch.logsumexp(negative_logits / self.temperature, dim=1)
        losses = -(positive_logits - negative_log_denominator)
        if self.reduction == "mean":
            return losses.mean()
        return losses.sum()


class WeightedSupervisedContrastiveOrdinalLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        num_classes: int = 5,
        margin_scale: float = 1.0,
        normalize_ordinal_distance: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.margin_scale = float(margin_scale)
        self.normalize_ordinal_distance = bool(normalize_ordinal_distance)
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        learnable_margins: CumulativeOrdinalMargins | None = None,
    ) -> torch.Tensor:
        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.long()
        batch_size = embeddings.shape[0]

        if batch_size < 3 or torch.unique(labels).numel() < 2:
            return embeddings.sum() * 0.0

        similarities = embeddings @ embeddings.T
        negative_mask = labels.unsqueeze(1) != labels.unsqueeze(0)
        self_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)

        if learnable_margins is None:
            distances = ordinal_distance(
                labels,
                labels,
                num_classes=self.num_classes,
                normalize=self.normalize_ordinal_distance,
                margin_scale=self.margin_scale,
            )
        else:
            distances = learnable_margins(labels, labels) * self.margin_scale
        negative_logits = similarities + negative_mask.float() * distances

        if sample_weights is None:
            sample_weights = torch.ones(batch_size, device=embeddings.device, dtype=embeddings.dtype)
        else:
            sample_weights = sample_weights.to(device=embeddings.device, dtype=embeddings.dtype)

        anchor_losses = []

        for anchor in range(batch_size):
            positive_indices = torch.where((labels == labels[anchor]) & (~self_mask[anchor]))[0]
            negative_indices = torch.where(negative_mask[anchor])[0]
            if positive_indices.numel() == 0 or negative_indices.numel() == 0:
                continue

            losses = []
            negative_log_denominator = torch.logsumexp(
                negative_logits[anchor, negative_indices] / self.temperature,
                dim=0,
            )
            for positive in positive_indices:
                positive_logit = similarities[anchor, positive] / self.temperature
                losses.append(-(positive_logit - negative_log_denominator))
            anchor_losses.append(torch.stack(losses).mean() * sample_weights[anchor])

        if not anchor_losses:
            return embeddings.sum() * 0.0

        losses = torch.stack(anchor_losses)
        if self.reduction == "mean":
            return losses.mean()
        return losses.sum()


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(predictions.float(), targets.float()) + eps)
