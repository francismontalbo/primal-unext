# PRIMAL-UNeXt
- Author: Dr. Francis Jesmar P. Montalbo
- Affiliation: Batangas State University
- Department: Center for Artificial Intelligence and Smart Technologies (CAIST) and College of Informatics and Computing Sciences (CICS)
- Email: francismontalbo@ieee.org, francisjesmar.montalbo@g.batstate-u.edu.ph
- Website: https://francismontalbo.github.io

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

**Note: The LiTS 2017 dataset had the dataset partitioned accordingly based on the manuscript and did not use the hidden test sets from Coda, similar to the other previously published research papers.**

- **LiTS (Liver Tumor Segmentation Challenge)**: https://competitions.codalab.org/competitions/17094
- **3D-IRCADb-01**: https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/

After download, convert to your standardized training layout (`images`, `liver_masks`, `tumor_masks`, `multi_class_masks`) before training.

## Notes

- Binary segmentation uses sigmoid output (`num_classes=1`).
- Multi-class segmentation uses softmax output (`num_classes>1`).
- The architecture is implemented in TensorFlow/Keras with class-style blocks for easier extension and testing.
