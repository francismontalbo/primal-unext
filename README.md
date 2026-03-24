# PRIMAL-UNeXt

Reference implementation package for the PRIMAL-UNeXt medical image segmentation model used in manuscript experiments.

## Repository goals

- Provide a clean, reusable model package separated from notebook-only experimentation.
- Keep architecture components modular (`layers`, `model`, `metrics`, `utils`) with paper-aligned blocks.
- Offer a one-command validation run to confirm model wiring locally.
- Support reviewer-period code sharing with privacy-aware defaults.

## Project layout

```text
.
├── PRIMAL_UNeXt.ipynb          # experiment notebook used during development
├── main.py                     # quick validation / smoke-test entrypoint
├── requirements.txt            # essential dependencies for setup
└── primal_unext/
    ├── __init__.py
    ├── layers.py               # custom layers (GRNS, AdaLNIN, ResAdd)
    ├── metrics.py              # segmentation metrics
    ├── model.py                # class-style PRIMAL-UNeXt model
    └── utils.py                # seed manager, inspectors, synthetic data tools
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick validation

```bash
python main.py --image-size 128 --channels 3 --num-classes 1
```

This command will:
- build the model,
- run a synthetic forward pass,
- run mini evaluation,
- print parameter and approximate memory statistics.

## Library usage

```python
from primal_unext import build_primal_unext

model = build_primal_unext(
    image_size=128,
    in_channels=3,
    num_classes=1,
)
```

## Dataset sources used in liver/tumor segmentation literature

The notebook workflow expects a preprocessed folder structure (`prepared_liver_dataset/...`).
For official raw datasets commonly used for this task, use:

- **LiTS (Liver Tumor Segmentation Challenge)**: https://competitions.codalab.org/competitions/17094
- **MSD Task03 Liver**: http://medicaldecathlon.com/
- **3D-IRCADb-01**: https://www.ircad.fr/research/3d-ircadb-01/

After download, convert to your standardized training layout (`images`, `liver_masks`, `tumor_masks`, `multi_class_masks`) before training.

## Reviewer-period privacy checklist

Before sharing artifacts generated from `PRIMAL_UNeXt.ipynb`, ensure:

1. Notebook outputs are cleared (plots, logs, local paths, timing traces).
2. Dataset paths do not expose private local directory structures.
3. Any exported tables include only aggregate metrics.
4. Saved checkpoints and logs are reviewed for embedded metadata.

## Notes

- Binary segmentation uses sigmoid output (`num_classes=1`).
- Multi-class segmentation uses softmax output (`num_classes>1`).
- The architecture is implemented in TensorFlow/Keras with class-style blocks for easier extension and testing.
