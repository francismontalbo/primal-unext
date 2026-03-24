# PRIMAL-UNeXt

Reference implementation package for the PRIMAL-UNeXt medical image segmentation model used in manuscript experiments.

## Repository goals

- Provide a clean, reusable model package separated from notebook-only experimentation.
- Keep architecture components modular (`layers`, `model`, `metrics`, `utils`).
- Offer a one-command smoke test to validate installation and model wiring.
- Support reviewer-period code sharing with privacy-aware defaults.

## Project layout

```text
.
├── PRIMAL_UNeXt.ipynb          # experiment notebook used during development
├── main.py                     # quick validation / smoke-test entrypoint
└── primal_unext/
    ├── __init__.py
    ├── layers.py               # custom layers (GRNS, AdaLNIN, ResAdd)
    ├── metrics.py              # segmentation metrics
    ├── model.py                # PRIMAL-UNeXt model builder
    └── utils.py                # configuration, seeding, stats, synthetic data
```

## Quick start

### 1) Install dependencies

```bash
pip install tensorflow
```

### 2) Run the validation script

```bash
python main.py --image-size 128 --channels 3 --num-classes 1
```

This script will:
- build the PRIMAL-UNeXt model,
- run a synthetic forward pass,
- run a mini evaluation,
- print parameter and size statistics.

## Library usage

```python
from primal_unext import build_primal_unext

model = build_primal_unext(
    input_shape=(128, 128, 3),
    num_classes=1,
    base_filters=32,
)
```

## Reviewer-period privacy checklist

Before sharing artifacts generated from `PRIMAL_UNeXt.ipynb`, ensure:

1. Notebook outputs are cleared (plots, logs, local paths, timing traces).
2. Dataset paths do not expose private local directory structures.
3. Any exported results tables include only aggregate metrics.
4. Saved checkpoints and logs are reviewed for embedded metadata.

A safe export workflow is to keep the notebook for reproducibility while using the package modules for distributable code.

## Notes

- Binary segmentation uses sigmoid output (`num_classes=1`).
- Multi-class segmentation uses softmax output (`num_classes>1`).
- The architecture is implemented in TensorFlow/Keras and organized for straightforward extension.
