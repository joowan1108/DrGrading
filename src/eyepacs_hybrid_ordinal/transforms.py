from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropBlackBorder:
    """Crop the mostly black border around fundus images."""

    def __init__(self, threshold: int = 7, padding: int = 8) -> None:
        self.threshold = int(threshold)
        self.padding = int(padding)

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"))
        mask = gray > self.threshold
        if not mask.any():
            return image

        y_coords, x_coords = np.where(mask)
        x_min = max(int(x_coords.min()) - self.padding, 0)
        y_min = max(int(y_coords.min()) - self.padding, 0)
        x_max = min(int(x_coords.max()) + self.padding + 1, image.width)
        y_max = min(int(y_coords.max()) + self.padding + 1, image.height)
        return image.crop((x_min, y_min, x_max, y_max))


def normalization_transform(mode: str) -> list:
    mode = mode.lower()
    if mode in {"none", "zero_one", "0_1"}:
        return []
    if mode == "imagenet":
        return [T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    raise ValueError(f"Unknown normalization mode '{mode}'. Use 'none' or 'imagenet'.")


def make_train_transform(image_size: int = 300, normalize: str = "none") -> T.Compose:
    return T.Compose(
        [
            CropBlackBorder(),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(
                degrees=10,
                interpolation=T.InterpolationMode.BICUBIC,
                fill=0,
            ),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            T.ToTensor(),
            *normalization_transform(normalize),
        ]
    )


def make_eval_transform(image_size: int = 300, normalize: str = "none") -> T.Compose:
    return T.Compose(
        [
            CropBlackBorder(),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            *normalization_transform(normalize),
        ]
    )
