from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def _build_efficientnet(name: str, pretrained_imagenet: bool) -> tuple[nn.Module, int]:
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision EfficientNet backbone '{name}'.")

    model_fn = getattr(models, name)
    try:
        model = model_fn(weights="DEFAULT" if pretrained_imagenet else None)
    except TypeError:
        model = model_fn(pretrained=pretrained_imagenet)

    if not hasattr(model, "classifier"):
        raise ValueError(f"Backbone '{name}' does not expose an EfficientNet-style classifier.")

    classifier = model.classifier
    if isinstance(classifier, nn.Sequential):
        linear_layers = [module for module in classifier.modules() if isinstance(module, nn.Linear)]
        feature_dim = linear_layers[-1].in_features
    elif isinstance(classifier, nn.Linear):
        feature_dim = classifier.in_features
    else:
        raise ValueError(f"Could not infer feature dimension from classifier: {classifier}.")

    model.classifier = nn.Identity()
    return model, feature_dim


def _build_resnet(name: str, pretrained_imagenet: bool) -> tuple[nn.Module, int]:
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision ResNet backbone '{name}'.")

    model_fn = getattr(models, name)
    try:
        model = model_fn(weights="DEFAULT" if pretrained_imagenet else None)
    except TypeError:
        model = model_fn(pretrained=pretrained_imagenet)

    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


def build_encoder(backbone: str, pretrained_imagenet: bool = False) -> tuple[nn.Module, int]:
    backbone = backbone.lower()
    if backbone.startswith("efficientnet"):
        return _build_efficientnet(backbone, pretrained_imagenet)
    if backbone.startswith("resnet"):
        return _build_resnet(backbone, pretrained_imagenet)
    raise ValueError("Supported backbones currently include torchvision EfficientNet and ResNet models.")


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1280, output_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1280, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class HybridOrdinalNet(nn.Module):
    """Backbone with PCOL/SCOLw projections and an ordinal regression head."""

    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        pretrained_imagenet: bool = False,
        projection_hidden_dim: int = 1280,
        projection_dim: int = 128,
        regression_hidden_dim: int = 1280,
        dropout: float = 0.0,
        regression_input: str = "backbone",
    ) -> None:
        super().__init__()
        if regression_input not in {"backbone", "projection_concat"}:
            raise ValueError(
                "regression_input must be either 'backbone' or 'projection_concat', "
                f"got {regression_input!r}."
            )

        self.encoder, self.feature_dim = build_encoder(backbone, pretrained_imagenet)
        self.regression_input = regression_input
        self.pcol_head = ProjectionHead(self.feature_dim, projection_hidden_dim, projection_dim, dropout=dropout)
        self.scol_head = ProjectionHead(self.feature_dim, projection_hidden_dim, projection_dim, dropout=dropout)
        regression_input_dim = 2 * projection_dim if regression_input == "projection_concat" else self.feature_dim
        self.regression_head = RegressionHead(regression_input_dim, regression_hidden_dim, dropout=dropout)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        with torch.amp.autocast(device_type=features.device.type, enabled=False):
            head_features = features.float()
            pcol_embedding = self.pcol_head(head_features)
            scol_embedding = self.scol_head(head_features)
            if self.regression_input == "projection_concat":
                regression_features = torch.cat((pcol_embedding, scol_embedding), dim=1)
            else:
                regression_features = head_features

            return {
                "features": head_features,
                "pcol": pcol_embedding,
                "scol": scol_embedding,
                "prediction": self.regression_head(regression_features),
            }


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, strict: bool = True) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(clean_state_dict_keys(state_dict), strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    return checkpoint
