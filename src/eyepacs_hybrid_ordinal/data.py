from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

from .splitting import build_split_indices


IMAGE_COLUMN_CANDIDATES = ("image", "id_code", "filename", "file", "path")
LABEL_COLUMN_CANDIDATES = ("level", "diagnosis", "label", "target", "grade")
SUBJECT_COLUMN_CANDIDATES = ("subject", "subject_id", "patient", "patient_id", "case", "case_id")
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")


def infer_column(columns: Iterable[str], candidates: Sequence[str], role: str) -> str:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    available = ", ".join(columns)
    expected = ", ".join(candidates)
    raise ValueError(f"Could not infer {role} column. Expected one of [{expected}], got [{available}].")


def infer_subject_id(image_id: str) -> str:
    stem = Path(str(image_id)).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem


def load_eyepacs_dataframe(
    labels_csv: str | Path,
    image_col: str | None = None,
    label_col: str | None = None,
    subject_col: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(labels_csv)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).copy()

    resolved_image_col = image_col or infer_column(frame.columns, IMAGE_COLUMN_CANDIDATES, "image id")
    resolved_label_col = label_col or infer_column(frame.columns, LABEL_COLUMN_CANDIDATES, "label")

    if subject_col is not None:
        resolved_subject_col = subject_col
    else:
        try:
            resolved_subject_col = infer_column(frame.columns, SUBJECT_COLUMN_CANDIDATES, "subject id")
        except ValueError:
            resolved_subject_col = None

    out = frame.copy()
    out["_image_id"] = out[resolved_image_col].astype(str)
    out["_label"] = out[resolved_label_col].astype(int)
    if resolved_subject_col is None:
        out["_subject_id"] = out["_image_id"].map(infer_subject_id)
    else:
        out["_subject_id"] = out[resolved_subject_col].astype(str)
    return out.reset_index(drop=True)


def resolve_image_path(image_root: str | Path, image_id: str) -> Path:
    root = Path(image_root)
    candidate = Path(str(image_id))

    if candidate.suffix:
        path = root / candidate
        if path.exists():
            return path
    else:
        for extension in IMAGE_EXTENSIONS:
            path = root / f"{image_id}{extension}"
            if path.exists():
                return path

    direct_path = root / str(image_id)
    if direct_path.exists():
        return direct_path

    raise FileNotFoundError(f"Could not find image '{image_id}' under '{root}'.")


class EyePACSDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        transform=None,
        verify_images: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_ids = self.frame["_image_id"].astype(str).tolist()
        self.targets = self.frame["_label"].astype(int).to_numpy()
        self._path_cache: dict[int, Path] = {}

        if verify_images:
            for index in range(len(self.image_ids)):
                self._path_cache[index] = resolve_image_path(self.image_root, self.image_ids[index])

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, index: int) -> Path:
        if index not in self._path_cache:
            self._path_cache[index] = resolve_image_path(self.image_root, self.image_ids[index])
        return self._path_cache[index]

    def __getitem__(self, index: int):
        image = Image.open(self._image_path(index))
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = int(self.targets[index])
        image_id = self.image_ids[index]
        return image, target, image_id


class ClassStratifiedBatchSampler(Sampler[list[int]]):
    """Guarantee class coverage, then fill each batch using the natural class distribution."""

    def __init__(
        self,
        targets: Sequence[int],
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 42,
        min_samples_per_class: int = 2,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning.")

        self.targets = np.asarray(targets).astype(int)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches or np.ceil(len(self.targets) / self.batch_size))
        self.seed = int(seed)
        self.min_samples_per_class = int(min_samples_per_class)
        self.epoch = 0

        self.class_to_indices = {
            int(label): np.flatnonzero(self.targets == label)
            for label in sorted(np.unique(self.targets).tolist())
        }
        self.active_classes = [label for label, values in self.class_to_indices.items() if len(values) > 0]
        if len(self.active_classes) < 2:
            raise ValueError("ClassStratifiedBatchSampler needs at least two classes.")
        if self.min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be at least 2 for supervised contrastive positives.")

        guaranteed_samples = self.min_samples_per_class * len(self.active_classes)
        if guaranteed_samples > self.batch_size:
            raise ValueError(
                "batch_size must fit min_samples_per_class samples for every active class: "
                f"need at least {guaranteed_samples}, got {self.batch_size}."
            )

        class_counts = np.asarray(
            [len(self.class_to_indices[label]) for label in self.active_classes],
            dtype=np.float64,
        )
        self.natural_class_probabilities = class_counts / class_counts.sum()

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        guaranteed_samples = self.min_samples_per_class * len(self.active_classes)
        natural_samples = self.batch_size - guaranteed_samples

        for _ in range(self.num_batches):
            samples_by_class = {
                label: self.min_samples_per_class
                for label in self.active_classes
            }
            if natural_samples > 0:
                sampled_labels = rng.choice(
                    self.active_classes,
                    size=natural_samples,
                    replace=True,
                    p=self.natural_class_probabilities,
                )
                for label in sampled_labels:
                    samples_by_class[int(label)] += 1

            batch: list[int] = []
            for label in self.active_classes:
                choices = self.class_to_indices[label]
                sample_count = samples_by_class[label]
                selected = rng.choice(
                    choices,
                    size=sample_count,
                    replace=len(choices) < sample_count,
                )
                batch.extend(selected.astype(int).tolist())

            rng.shuffle(batch)
            yield batch


def inverse_frequency_class_weights(targets: Sequence[int], num_classes: int) -> np.ndarray:
    targets_array = np.asarray(targets).astype(int)
    counts = np.bincount(targets_array, minlength=num_classes).astype(float)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights
