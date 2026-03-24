"""Quick validation entrypoint for PRIMAL-UNeXt.

Example:
    python main.py --image-size 128 --channels 3 --num-classes 1
"""

from __future__ import annotations

import argparse

import tensorflow as tf

from primal_unext import (
    ExperimentConfig,
    ModelInspector,
    SeedManager,
    SyntheticDataFactory,
    build_primal_unext,
    dice_binary,
    iou_binary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRIMAL-UNeXt quick validation")
    parser.add_argument("--image-size", type=int, default=128, help="Input image height and width")
    parser.add_argument("--channels", type=int, default=3, help="Input channels")
    parser.add_argument("--num-classes", type=int, default=1, help="Output segmentation classes")
    parser.add_argument("--batch-size", type=int, default=4, help="Synthetic batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SeedManager.set(args.seed)

    config = ExperimentConfig(
        image_height=args.image_size,
        image_width=args.image_size,
        channels=args.channels,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )

    model = build_primal_unext(
        image_size=args.image_size,
        in_channels=args.channels,
        num_classes=config.num_classes,
    )

    if config.num_classes == 1:
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_binary, iou_binary])
    else:
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    x, y = SyntheticDataFactory.make_batch(config)
    y_pred = model(x, training=False)
    evaluation = model.evaluate(x, y, verbose=0, return_dict=True)

    stats = ModelInspector.stats(model)

    print(f"Model: {model.name}")
    print(f"Input shape: {config.input_shape}")
    print(f"Output shape: {tuple(y_pred.shape)}")
    for key, value in stats.items():
        if "params" in key:
            print(f"{key}: {int(value):,}")
        else:
            print(f"{key}: {value:.2f}")

    print("Evaluation:")
    for key, value in evaluation.items():
        print(f"  {key}: {float(value):.4f}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
