"""Regression attribution and pooling-moment maps; no training or model changes."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def positive_unit_map(values: np.ndarray) -> np.ndarray:
    """Display positive evidence on [0,1]; a zero map remains exactly zero."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite two-dimensional attribution map.")
    positive = values.clip(min=0)
    maximum = float(positive.max())
    return positive / maximum if maximum > 0 else np.zeros_like(positive)


def pooling_moment_map(features: torch.Tensor, exponent: float, eps=1e-6) -> np.ndarray:
    """Mean of channel-wise x**p/sum(x**p), on one fixed nonnegative feature map.

    This describes the normalized powered sum inside pooling, not the final
    regression prediction. Log-space softmax avoids overflow at large p.
    """
    if features.ndim != 3 or not torch.isfinite(features).all():
        raise ValueError("Expected finite CxHxW features.")
    if not np.isfinite(exponent) or exponent <= 0 or eps <= 0:
        raise ValueError("The pooling exponent and epsilon must be positive.")
    channels, height, width = features.shape
    logits = features.detach().to(dtype=torch.float64).clamp_min(eps).log()
    weights = (float(exponent) * logits.reshape(channels, -1)).softmax(dim=1)
    return weights.mean(dim=0).reshape(height, width).cpu().numpy().astype(np.float32)


def regression_maps(model, image: torch.Tensor) -> tuple[float, dict[str, np.ndarray]]:
    """Differentiate the unrounded, unclamped scalar severity, in evaluation mode.

    Grad-CAM averages gradients spatially. Gradient x activation also retains
    their location dependence, which matters for a nonlinear spatial pool.
    The model parameters are frozen; gradients are enabled on the input only.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Use an eval-mode model with frozen parameters.")
    captured = {}

    def capture(_module, _inputs, output):
        captured["features"] = output

    hook = model.encoder.features.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            outputs = model(image.detach().unsqueeze(0).requires_grad_(True))
            prediction = outputs["prediction"]
            if prediction.numel() != 1 or not torch.isfinite(prediction).all():
                raise ValueError("Expected one finite regression prediction.")
            features = captured["features"]
            if features.ndim != 4 or features.shape[0] != 1:
                raise ValueError("Expected one CxHxW convolutional feature map.")
            gradients, = torch.autograd.grad(prediction.sum(), features)
        if not torch.isfinite(features).all() or not torch.isfinite(gradients).all():
            raise ValueError("Non-finite convolutional activations or gradients.")
        features, gradients = features.detach()[0], gradients.detach()[0]
        gradcam = (gradients.mean(dim=(-2, -1), keepdim=True) * features).sum(dim=0)
        local = (gradients * features).sum(dim=0)
        maps = {"gradcam_signed": gradcam.cpu().numpy(),
                "gradient_activation_signed": local.cpu().numpy()}
        for name in ["gradcam", "gradient_activation"]:
            maps[name] = np.maximum(maps[name + "_signed"], 0)
        if model.global_pool is not None:
            p = float(model.global_pool.p.detach().clamp_min(model.global_pool.eps))
            maps["moment_gap"] = pooling_moment_map(features, 1.0, model.global_pool.eps)
            maps["moment_gem"] = pooling_moment_map(features, p, model.global_pool.eps)
        return float(prediction.detach().item()), maps
    finally:
        hook.remove()


def resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))[None, None]
    return F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False)[0, 0].numpy()


def lesion_map_metrics(values: np.ndarray, lesion_union: np.ndarray | None,
                       crop: tuple[int, int, int, int]) -> dict:
    """Mass and area enrichment in true mask pixels in the model-visible crop.

    Display-only bounding-box padding is deliberately irrelevant here.
    A missing mask and an empty positive attribution are recorded as undefined.
    """
    values = np.maximum(np.asarray(values, dtype=np.float32), 0)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite two-dimensional map.")
    result = {"positive_map_max": float(values.max()),
              "has_positive_mass": bool(values.max() > 0),
              "lesion_area_fraction": None, "lesion_mass_fraction": None,
              "lesion_area_enrichment": None}
    if lesion_union is None:
        return result
    x0, y0, x1, y1 = crop
    mask = np.asarray(lesion_union[y0:y1, x0:x1], dtype=bool)
    if mask.shape != (y1-y0, x1-x0) or mask.size == 0:
        raise ValueError("Lesion mask does not cover the actual inference crop.")
    area = float(mask.mean())
    result["lesion_area_fraction"] = area
    expanded = resize_map(values, *mask.shape)
    mass = float(expanded.sum(dtype=np.float64))
    if mass > 0 and area > 0:
        fraction = float(expanded[mask].sum(dtype=np.float64) / mass)
        result.update(lesion_mass_fraction=fraction, lesion_area_enrichment=fraction/area)
    return result
