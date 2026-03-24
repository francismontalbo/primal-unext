"""Quick validation entrypoint for PRIMAL-UNeXt.

Example:
    python main.py --image-size 128 --channels 3 --num-classes 1
"""

from __future__ import annotations

import argparse

import tensorflow as tf

from primal_unext import (
    ExperimentConfig,
    build_primal_unext,
    compute_model_stats,
    dice_binary,
    iou_binary,
    set_global_seed,
    synthetic_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRIMAL-UNeXt model smoke test")
    parser.add_argument("--image-size", type=int, default=128, help="Input image height and width")
    parser.add_argument("--channels", type=int, default=3, help="Input channels")
    parser.add_argument("--num-classes", type=int, default=1, help="Output segmentation classes")
    parser.add_argument("--batch-size", type=int, default=4, help="Synthetic batch size")
    parser.add_argument("--base-filters", type=int, default=32, help="Base filter width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    cfg = ExperimentConfig(
        image_height=args.image_size,
        image_width=args.image_size,
        channels=args.channels,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )

    model = build_primal_unext(
        input_shape=cfg.input_shape,
        num_classes=cfg.num_classes,
        base_filters=args.base_filters,
    )

    if cfg.num_classes == 1:
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_binary, iou_binary])
    else:
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    x, y = synthetic_batch(cfg)
    out = model(x, training=False)
    eval_results = model.evaluate(x, y, verbose=0, return_dict=True)

    stats = compute_model_stats(model)

    print(f"Model: {model.name}")
    print(f"Input shape: {cfg.input_shape}")
    print(f"Output shape: {tuple(out.shape)}")
    for key, value in stats.items():
        if "params" in key:
            print(f"{key}: {int(value):,}")
        else:
            print(f"{key}: {value:.2f}")

    print("Evaluation:")
    for key, value in eval_results.items():
        print(f"  {key}: {float(value):.4f}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
