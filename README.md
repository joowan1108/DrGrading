# EyePACS Contrastive Learning with ResNet

This is a PyTorch starter project for contrastive representation learning on the
EyePACS diabetic retinopathy dataset. It uses a ResNet encoder with a SimCLR
projection head, then evaluates the learned features with a linear classifier.

## Project Layout

```text
.
├── configs/
│   ├── linear_eval_resnet18.yaml
│   └── simclr_resnet18_eyepacs.yaml
├── scripts/
│   ├── linear_eval.py
│   └── pretrain_simclr.py
└── src/
    └── eyepacs_contrastive/
        ├── data.py
        ├── losses.py
        ├── models.py
        ├── transforms.py
        └── utils.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Dataset

Download EyePACS from Kaggle after accepting the dataset terms, then place it
like this:

```text
data/
└── eyepacs/
    ├── trainLabels.csv
    └── train/
        ├── 10_left.jpeg
        ├── 10_right.jpeg
        └── ...
```

The loader expects an image id column such as `image` or `id_code`, and a label
column such as `level` or `diagnosis`. You can override the column names in the
YAML configs if needed.

## SimCLR Pretraining

```powershell
python scripts/pretrain_simclr.py --config configs/simclr_resnet18_eyepacs.yaml
```

Checkpoints are written to `checkpoints/simclr_resnet18/`.

## Linear Evaluation

After pretraining, run:

```powershell
python scripts/linear_eval.py --config configs/linear_eval_resnet18.yaml
```

The linear evaluation script freezes the ResNet encoder by default and trains a
single classifier layer for the 5 EyePACS severity grades.

## Practical Notes

- EyePACS is large. Start with `data.max_samples` in the config for a quick
  smoke test, then set it to `null` for full training.
- Contrastive learning benefits from large batches. If your GPU is memory
  limited, use ResNet-18, lower `data.image_size`, or use gradient accumulation
  as a later extension.
- The augmentations include fundus border cropping, random resized crops,
  flips, color jitter, grayscale, and Gaussian blur.
