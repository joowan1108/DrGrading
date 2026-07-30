from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CumulativeOrdinalMargins(nn.Module):
    """Learn independent bounded adjacent margins and accumulate them across ranks."""

    def __init__(
        self,
        num_classes: int,
        initial_margin: float = 0.5,
        minimum_margin: float = 0.0,
        initial_margin_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if not 0 <= minimum_margin < initial_margin < 1:
            raise ValueError(
                "margins must satisfy 0 <= minimum_margin < initial_margin < 1."
            )
        if initial_margin_jitter < 0:
            raise ValueError("initial_margin_jitter must be non-negative.")

        self.num_classes = int(num_classes)
        self.minimum_margin = float(minimum_margin)
        lower = max(
            self.minimum_margin + 1e-6,
            float(initial_margin) - float(initial_margin_jitter),
        )
        upper = min(1.0 - 1e-6, float(initial_margin) + float(initial_margin_jitter))
        if lower >= upper:
            raise ValueError("initial margin jitter produces an empty interval.")
        if initial_margin_jitter > 0:
            initial_values = torch.empty(self.num_classes - 1).uniform_(lower, upper)
        else:
            initial_values = torch.full(
                (self.num_classes - 1,),
                float(initial_margin),
            )
        scaled_initial_values = (
            (initial_values - self.minimum_margin)
            / (1.0 - self.minimum_margin)
        )
        self.raw_margins = nn.Parameter(torch.logit(scaled_initial_values))

    def margin_values(self) -> torch.Tensor:
        return self.minimum_margin + (
            (1.0 - self.minimum_margin) * torch.sigmoid(self.raw_margins)
        )

    def class_positions(self, normalize_mean: bool = False) -> torch.Tensor:
        margins = self.margin_values()
        if normalize_mean:
            margins = margins / margins.mean().clamp_min(1e-12)
        zero = self.raw_margins.new_zeros(1)
        return torch.cat((zero, torch.cumsum(margins, dim=0)))

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        normalize_mean: bool = False,
    ) -> torch.Tensor:
        left = left.long()
        right = right.long()
        positions = self.class_positions(normalize_mean=normalize_mean)
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
        detach_learnable_margins: bool = False,
        normalize_learnable_margins: bool = False,
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
            distances = learnable_margins(
                labels,
                prototype_labels,
                normalize_mean=normalize_learnable_margins,
            ) * self.margin_scale
            if detach_learnable_margins:
                distances = distances.detach()

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
        objective: str = "logsumexp",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        if objective not in {"logsumexp", "mmnp"}:
            raise ValueError("objective must be 'logsumexp' or 'mmnp'.")
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.margin_scale = float(margin_scale)
        self.normalize_ordinal_distance = bool(normalize_ordinal_distance)
        self.reduction = reduction
        self.objective = objective
        self.last_active_violation_rate: float | None = None
        self.last_num_comparisons = 0
        self.last_boundary_stats: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        learnable_margins: CumulativeOrdinalMargins | None = None,
        detach_learnable_margins: bool = False,
        normalize_learnable_margins: bool = False,
    ) -> torch.Tensor:
        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.long()
        batch_size = embeddings.shape[0]

        if batch_size < 3 or torch.unique(labels).numel() < 2:
            self.last_active_violation_rate = None
            self.last_num_comparisons = 0
            self.last_boundary_stats = None
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
            distances = learnable_margins(
                labels,
                labels,
                normalize_mean=normalize_learnable_margins,
            ) * self.margin_scale
            if detach_learnable_margins:
                distances = distances.detach()
        if sample_weights is None:
            sample_weights = torch.ones(batch_size, device=embeddings.device, dtype=embeddings.dtype)
        else:
            sample_weights = sample_weights.to(device=embeddings.device, dtype=embeddings.dtype)

        anchor_losses = []
        anchor_weights = []
        active_violations = embeddings.new_zeros((), dtype=torch.long)
        total_violations = 0
        num_boundaries = self.num_classes - 1
        boundary_comparisons = embeddings.new_zeros(num_boundaries)
        boundary_active = embeddings.new_zeros(num_boundaries)
        boundary_loss_sum = embeddings.new_zeros(num_boundaries)
        boundary_weighted_comparisons = embeddings.new_zeros(num_boundaries)
        boundary_weighted_active = embeddings.new_zeros(num_boundaries)
        boundary_weighted_loss_sum = embeddings.new_zeros(num_boundaries)

        if self.objective == "logsumexp":
            negative_logits = similarities + negative_mask.float() * distances

        for anchor in range(batch_size):
            positive_indices = torch.where((labels == labels[anchor]) & (~self_mask[anchor]))[0]
            negative_indices = torch.where(negative_mask[anchor])[0]
            if positive_indices.numel() == 0 or negative_indices.numel() == 0:
                continue

            if self.objective == "mmnp":
                positive_similarities = similarities[anchor, positive_indices]
                negative_similarities = similarities[anchor, negative_indices]
                negative_margins = distances[anchor, negative_indices]
                violations = (
                    negative_margins.unsqueeze(0)
                    + negative_similarities.unsqueeze(0)
                    - positive_similarities.unsqueeze(1)
                )
                hinge_losses = F.relu(violations)
                anchor_losses.append(hinge_losses.mean())
                active_violations = active_violations + (violations > 0).sum()
                total_violations += violations.numel()
                negative_labels = labels[negative_indices]
                anchor_label = labels[anchor]
                anchor_weight = sample_weights[anchor].detach()
                for boundary in range(num_boundaries):
                    crosses_boundary = (
                        ((anchor_label <= boundary) & (negative_labels > boundary))
                        | ((anchor_label > boundary) & (negative_labels <= boundary))
                    )
                    boundary_violations = violations[:, crosses_boundary]
                    boundary_hinge_losses = hinge_losses[:, crosses_boundary]
                    comparison_count = boundary_violations.numel()
                    active_count = (boundary_violations > 0).sum()
                    detached_loss_sum = boundary_hinge_losses.detach().sum()
                    boundary_comparisons[boundary] += comparison_count
                    boundary_active[boundary] += active_count
                    boundary_loss_sum[boundary] += detached_loss_sum
                    boundary_weighted_comparisons[boundary] += (
                        anchor_weight * comparison_count
                    )
                    boundary_weighted_active[boundary] += (
                        anchor_weight * active_count
                    )
                    boundary_weighted_loss_sum[boundary] += (
                        anchor_weight * detached_loss_sum
                    )
            else:
                losses = []
                negative_log_denominator = torch.logsumexp(
                    negative_logits[anchor, negative_indices] / self.temperature,
                    dim=0,
                )
                for positive in positive_indices:
                    positive_logit = similarities[anchor, positive] / self.temperature
                    losses.append(-(positive_logit - negative_log_denominator))
                anchor_losses.append(torch.stack(losses).mean())
            anchor_weights.append(sample_weights[anchor])

        if not anchor_losses:
            self.last_active_violation_rate = None
            self.last_num_comparisons = 0
            self.last_boundary_stats = None
            return embeddings.sum() * 0.0

        losses = torch.stack(anchor_losses)
        weights = torch.stack(anchor_weights)
        if self.objective == "mmnp":
            self.last_active_violation_rate = (
                active_violations.detach().float().item() / total_violations
            )
            self.last_num_comparisons = total_violations
            self.last_boundary_stats = {
                "comparisons": boundary_comparisons.detach(),
                "active": boundary_active.detach(),
                "loss_sum": boundary_loss_sum.detach(),
                "weighted_comparisons": boundary_weighted_comparisons.detach(),
                "weighted_active": boundary_weighted_active.detach(),
                "weighted_loss_sum": boundary_weighted_loss_sum.detach(),
            }
        else:
            self.last_active_violation_rate = None
            self.last_num_comparisons = 0
            self.last_boundary_stats = None
        if self.reduction == "mean":
            return (losses * weights).sum() / weights.sum().clamp_min(1e-12)
        return (losses * weights).sum()


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(predictions.float(), targets.float()) + eps)
