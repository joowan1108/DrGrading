# EyePACS Hybrid Contrastive Ordinal Regression

This project implements the method from `hybrid.pdf`: a hybrid supervised
contrastive ordinal regression framework for disease severity grading on
imbalanced medical datasets.

The training objective combines:

- `PCOL`: prototype-based contrastive ordinal loss.
- `SCOLw`: weighted supervised contrastive ordinal loss.
- `RMSE`: regression loss for ordinal disease grade prediction.

The default config follows the paper setting for diabetic retinopathy:
EfficientNet-V2S, 300 x 300 images, batch size 24, 75 epochs, class-stratified
batches, ReduceLROnPlateau, and early stopping.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## EyePACS Layout

Download EyePACS after accepting the Kaggle terms, then place it like this:

```text
data/
+-- eyepacs/
    +-- trainLabels.csv
    +-- train/
        +-- 10_left.jpeg
        +-- 10_right.jpeg
        +-- ...
```

The CSV should contain an image id column such as `image` or `id_code`, and an
ordinal label column such as `level` or `diagnosis`. The loader infers these by
default, or you can set explicit column names in the YAML config.

## Train One Fold

```powershell
python scripts/train_hybrid_ordinal.py --config configs/hybrid_eyepacs_efficientnet_v2_s.yaml
```

That trains the fold selected by `split.fold_index`.

## Run 10-Fold Cross-Validation

```powershell
python scripts/run_cross_validation.py --config configs/hybrid_eyepacs_efficientnet_v2_s.yaml
```

Each fold is written under `checkpoints/hybrid_eyepacs_efficientnet_v2_s/fold_*`.
The aggregate file is `cross_validation_metrics.json`.

## Evaluate A Checkpoint

```powershell
python scripts/evaluate_hybrid_ordinal.py `
  --config configs/hybrid_eyepacs_efficientnet_v2_s.yaml `
  --checkpoint checkpoints/hybrid_eyepacs_efficientnet_v2_s/best.pt
```

## Quick Smoke Test

For a fast local test, set `data.max_samples` to a small value in the config.
Use a value large enough to include at least two samples per class, otherwise
contrastive positives may be missing for rare classes.

## Paper-Faithful Defaults

- `PCOL` and `SCOLw` follow Eq. (1) and Eq. (2) literally: raw dot products,
  negative-only denominators, class inverse-frequency weights for `SCOLw`, and
  unnormalized scalar label distance by default.
- The default transform follows the PDF implementation details: resize to
  `300 x 300`, convert to tensor, and keep pixel values in `[0, 1]`.
- The model uses a shared EfficientNet-V2S encoder with three parallel heads:
  two projection MLPs with dense layers `1280 -> 128`, and one RMSE-optimized
  regression head used for inference.
