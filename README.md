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

That reserves the outer fold selected by `split.fold_index` for testing. The
remaining subjects are split into training and validation sets using
`split.val_size`; only the inner validation set controls the LR scheduler,
early stopping, and best-checkpoint selection. After training, `best.pt` is
evaluated once on the untouched outer test fold.

## Run 10-Fold Cross-Validation

```powershell
python scripts/run_cross_validation.py --config configs/hybrid_eyepacs_efficientnet_v2_s.yaml
```

Each fold is written under the configured output directory in `fold_*`. The
aggregate `cross_validation_metrics.json` contains outer-test metrics, not the
inner-validation metrics used for model selection. It also reports overall and
per-class correct, adjacent-error (`|prediction - target| = 1`), and
non-adjacent-error (`|prediction - target| > 1`) rates for comparison with the
paper's stacked-bar analysis. Directional metrics define underdiagnosis as
`prediction < target` and overdiagnosis as `prediction > target`; rates and
mean class distances are reported overall and per class for every outer fold.

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

- `PCOL` and `SCOLw` follow Eq. (1) and Eq. (2): negative-only denominators,
  dynamic batch-level inverse-frequency sample weights for `SCOLw`, and
  unnormalized scalar label distance by default.
- The default transform follows the PDF implementation details: resize to
  `300 x 300`, convert to tensor, and keep pixel values in `[0, 1]`.
- The model uses a shared EfficientNet-V2S encoder and two projection MLPs with
  dense layers `1280 -> 128`. Their embeddings are concatenated into a
  256-dimensional representation and passed to the RMSE-optimized regression
  head, so all three objectives are optimized through one connected forward path.
