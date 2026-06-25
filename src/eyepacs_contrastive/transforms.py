from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropBlackBorder:
    """Crop the mostly black border around fundus images."""

    def __init__(self, threshold: int = 7, padding: int = 8) -> None:
        self.threshold = threshold
        self.padding = padding

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


def _blur_kernel_size(image_size: int) -> int:
    kernel = max(3, int(0.1 * image_size))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def make_simclr_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            CropBlackBorder(),
            T.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.15),
            T.RandomApply([T.GaussianBlur(kernel_size=_blur_kernel_size(image_size), sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_supervised_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            CropBlackBorder(),
            T.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)],
                p=0.5,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_eval_transform(image_size: int = 224) -> T.Compose:
    resize_size = int(round(image_size * 1.14))
    return T.Compose(
        [
            CropBlackBorder(),
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
