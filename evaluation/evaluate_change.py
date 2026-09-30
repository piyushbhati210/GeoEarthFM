import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_mask(path):
    mask = np.load(path).astype(np.float32)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, but got shape: {mask.shape}"
        )

    return mask


def calculate_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "Dice": float(dice),
        "IoU": float(iou),
        "Accuracy": float(accuracy),
    }


def save_visualization(
    probability,
    prediction,
    ground_truth,
    output_path
):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(probability, cmap="gray")
    plt.title("Prediction Probability")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Model Prediction")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def save_error_map(prediction, ground_truth, output_path):
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    error_map = np.zeros(
        pred.shape,
        dtype=np.uint8
    )

    error_map[
        np.logical_and(pred, ~gt)
    ] = 1

    error_map[
        np.logical_and(~pred, gt)
    ] = 2

    error_map[
        np.logical_and(pred, gt)
    ] = 3

    Image.fromarray(error_map).save(output_path)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate STTORM change detection prediction"
    )

    parser.add_argument(
        "--prediction",
        required=True
    )

    parser.add_argument(
        "--ground-truth",
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--out",
        default="outputs/change_evaluation"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("STTORM CHANGE DETECTION EVALUATION")
    print("=" * 65)

    prediction_path = Path(args.prediction)
    ground_truth_path = Path(args.ground_truth)
    output_prefix = Path(args.out)

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{prediction_path}"
        )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n{ground_truth_path}"
        )

    print()
    print("Loading prediction:")
    print(prediction_path)

    probability = np.load(
        prediction_path
    ).astype(np.float32)

    probability = np.squeeze(probability)

    print()
    print("Loading ground truth:")
    print(ground_truth_path)

    ground_truth_raw = load_mask(
        ground_truth_path
    )

    print()
    print("Prediction shape :", probability.shape)
    print("Ground truth shape:", ground_truth_raw.shape)

    if probability.shape != ground_truth_raw.shape:
        raise ValueError(
            "\nShape mismatch!\n"
            f"Prediction: {probability.shape}\n"
            f"Ground truth: {ground_truth_raw.shape}\n"
        )

    ground_truth = ground_truth_raw > 0

    prediction = probability >= args.threshold

    metrics = calculate_metrics(
        prediction,
        ground_truth
    )

    total_pixels = probability.size

    predicted_change_pixels = int(
        prediction.sum()
    )

    ground_truth_change_pixels = int(
        ground_truth.sum()
    )

    predicted_change_percent = (
        100.0 *
        predicted_change_pixels /
        total_pixels
    )

    ground_truth_change_percent = (
        100.0 *
        ground_truth_change_pixels /
        total_pixels
    )

    print()
    print("=" * 65)
    print("CHANGE DETECTION RESULTS")
    print("=" * 65)

    print()
    print(f"Threshold              : {args.threshold}")
    print(f"Total pixels           : {total_pixels:,}")

    print()
    print("GROUND TRUTH")
    print("-" * 65)
    print(
        f"Changed pixels         : "
        f"{ground_truth_change_pixels:,}"
    )
    print(
        f"Change area            : "
        f"{ground_truth_change_percent:.2f}%"
    )

    print()
    print("MODEL PREDICTION")
    print("-" * 65)
    print(
        f"Changed pixels         : "
        f"{predicted_change_pixels:,}"
    )
    print(
        f"Change area            : "
        f"{predicted_change_percent:.2f}%"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 65)
    print(f"True Positive (TP)     : {metrics['TP']:,}")
    print(f"True Negative (TN)     : {metrics['TN']:,}")
    print(f"False Positive (FP)    : {metrics['FP']:,}")
    print(f"False Negative (FN)    : {metrics['FN']:,}")

    print()
    print("PERFORMANCE METRICS")
    print("-" * 65)
    print(f"Accuracy               : {metrics['Accuracy']:.4f}")
    print(f"Precision              : {metrics['Precision']:.4f}")
    print(f"Recall                 : {metrics['Recall']:.4f}")
    print(f"Dice / F1              : {metrics['Dice']:.4f}")
    print(f"IoU                    : {metrics['IoU']:.4f}")

    prediction_png = Path(
        str(output_prefix) + "_prediction.png"
    )

    ground_truth_png = Path(
        str(output_prefix) + "_ground_truth.png"
    )

    comparison_path = Path(
        str(output_prefix) + "_comparison.png"
    )

    error_path = Path(
        str(output_prefix) + "_error_map.png"
    )

    metrics_path = Path(
        str(output_prefix) + "_metrics.txt"
    )

    Image.fromarray(
        prediction.astype(np.uint8) * 255
    ).save(prediction_png)

    Image.fromarray(
        ground_truth.astype(np.uint8) * 255
    ).save(ground_truth_png)

    save_visualization(
        probability,
        prediction,
        ground_truth,
        comparison_path
    )

    save_error_map(
        prediction,
        ground_truth,
        error_path
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STTORM CHANGE DETECTION EVALUATION\n"
        )
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Prediction: {prediction_path}\n"
        )

        f.write(
            f"Ground truth: {ground_truth_path}\n"
        )

        f.write(
            f"Threshold: {args.threshold}\n\n"
        )

        f.write("GROUND TRUTH\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Changed pixels: "
            f"{ground_truth_change_pixels}\n"
        )
        f.write(
            f"Change area: "
            f"{ground_truth_change_percent:.2f}%\n\n"
        )

        f.write("PREDICTION\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Changed pixels: "
            f"{predicted_change_pixels}\n"
        )
        f.write(
            f"Change area: "
            f"{predicted_change_percent:.2f}%\n\n"
        )

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write(f"TP: {metrics['TP']}\n")
        f.write(f"TN: {metrics['TN']}\n")
        f.write(f"FP: {metrics['FP']}\n")
        f.write(f"FN: {metrics['FN']}\n\n")

        f.write("METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Accuracy: {metrics['Accuracy']:.4f}\n"
        )
        f.write(
            f"Precision: {metrics['Precision']:.4f}\n"
        )
        f.write(
            f"Recall: {metrics['Recall']:.4f}\n"
        )
        f.write(
            f"Dice / F1: {metrics['Dice']:.4f}\n"
        )
        f.write(
            f"IoU: {metrics['IoU']:.4f}\n"
        )

    print()
    print("=" * 65)
    print("FILES SAVED")
    print("=" * 65)

    print(prediction_png)
    print(ground_truth_png)
    print(comparison_path)
    print(error_path)
    print(metrics_path)

    print()
    print("=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()