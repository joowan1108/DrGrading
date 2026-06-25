from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def build_resnet_encoder(backbone: str = "resnet18", pretrained_imagenet: bool = False) -> tuple[nn.Module, int]:
    if not hasattr(models, backbone):
        raise ValueError(f"Unknown torchvision ResNet backbone '{backbone}'.")

    model_fn = getattr(models, backbone)
    try:
        resnet = model_fn(weights="DEFAULT" if pretrained_imagenet else None)
    except TypeError:
        resnet = model_fn(pretrained=pretrained_imagenet)

    feature_dim = resnet.fc.in_features
    resnet.fc = nn.Identity()
    return resnet, feature_dim


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SimCLRResNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained_imagenet: bool = False,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_resnet_encoder(backbone, pretrained_imagenet)
        self.feature_dim = feature_dim
        self.projector = ProjectionHead(feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        projections = self.projector(features)
        return features, projections


class LinearEvalModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 5,
        pretrained_imagenet: bool = False,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_resnet_encoder(backbone, pretrained_imagenet)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.classifier(features)


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_simclr_encoder(model: LinearEvalModel, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = clean_state_dict_keys(state_dict)

    encoder_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder.") :]] = value

    if not encoder_state:
        raise ValueError(f"No encoder weights found in checkpoint '{checkpoint_path}'.")

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected encoder checkpoint keys: {unexpected}")
    classifier_missing = [key for key in missing if key.startswith("fc.")]
    if classifier_missing:
        raise ValueError(f"Checkpoint appears incompatible with this backbone: missing {classifier_missing}")
