from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_COLUMN_CANDIDATES = ("image", "id_code", "filename", "file", "path")
LABEL_COLUMN_CANDIDATES = ("level", "diagnosis", "label", "target", "grade")
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")


def infer_column(columns: Iterable[str], candidates: Sequence[str], role: str) -> str:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    available = ", ".join(columns)
    expected = ", ".join(candidates)
    raise ValueError(f"Could not infer {role} column. Expected one of [{expected}], got [{available}].")


def resolve_image_path(image_root: str | Path, image_id: str) -> Path:
    root = Path(image_root)
    image_id = str(image_id)
    candidate = Path(image_id)

    if candidate.suffix:
        path = root / candidate
        if path.exists():
            return path
    else:
        for extension in IMAGE_EXTENSIONS:
            path = root / f"{image_id}{extension}"
            if path.exists():
                return path

    direct_path = root / image_id
    if direct_path.exists():
        return direct_path

    raise FileNotFoundError(f"Could not find image '{image_id}' under '{root}'.")


class EyePACSDataset(Dataset):
    """EyePACS dataset that can return one supervised view or two contrastive views."""

    def __init__(
        self,
        labels_csv: str | Path,
        image_root: str | Path,
        transform=None,
        image_col: str | None = None,
        label_col: str | None = None,
        max_samples: int | None = None,
        contrastive: bool = False,
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.image_root = Path(image_root)
        self.transform = transform
        self.contrastive = contrastive

        frame = pd.read_csv(self.labels_csv)
        if max_samples is not None:
            frame = frame.head(int(max_samples)).copy()

        self.image_col = image_col or infer_column(frame.columns, IMAGE_COLUMN_CANDIDATES, "image id")
        self.label_col = label_col or infer_column(frame.columns, LABEL_COLUMN_CANDIDATES, "label")

        self.frame = frame.reset_index(drop=True)
        self.image_ids = self.frame[self.image_col].astype(str).tolist()
        self.targets = self.frame[self.label_col].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, index: int) -> Image.Image:
        image_path = resolve_image_path(self.image_root, self.image_ids[index])
        image = Image.open(image_path)
        return ImageOps.exif_transpose(image).convert("RGB")

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = int(self.targets[index])
        image_id = self.image_ids[index]

        if self.contrastive:
            if self.transform is None:
                raise ValueError("A transform is required when contrastive=True.")
            view1 = self.transform(image)
            view2 = self.transform(image.copy())
            return view1, view2, target, image_id

        if self.transform is not None:
            image = self.transform(image)
        return image, target, image_id


def make_train_val_indices(targets: Sequence[int], val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(targets))
    targets_array = np.asarray(targets)

    try:
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            stratify=targets_array,
        )
    except (ImportError, ValueError):
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        split = int(round(len(shuffled) * (1.0 - val_size)))
        train_idx, val_idx = shuffled[:split], shuffled[split:]

    return np.asarray(train_idx), np.asarray(val_idx)
