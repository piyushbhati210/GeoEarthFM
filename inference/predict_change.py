import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geoearthfm.config import *
from geoearthfm.models import ChangeDetector
from geoearthfm.utils import device, load_checkpoint, normalize_spectral, to_chw


# ---------------------------------------------------------
# Load one STTORM .npy image
# ---------------------------------------------------------
def prepare(path, dev):
    x = np.load(path).astype(np.float32)

    # Keep the 10 official STTORM spectral bands
    x = x[:, :, STTORM_BAND_INDICES]

    # Normalize spectral values
    x = normalize_spectral(x)

    # HWC -> CHW -> batch
    tensor = torch.from_numpy(to_chw(x)).unsqueeze(0)

    # Move to GPU
    return tensor.to(dev, non_blocking=True)


# ---------------------------------------------------------
# Predict change
# ---------------------------------------------------------
def predict(model, before, after, dev):
    before = prepare(before, dev)
    after = prepare(after, dev)

    with torch.inference_mode():
        logits = model(before, after)
        probability = torch.sigmoid(logits)

    return probability[0, 0].float().cpu().numpy()


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------
def save_outputs(probability, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Binary mask
    mask = (probability >= 0.5).astype(np.uint8) * 255

    mask_path = output_prefix.with_name(
        output_prefix.name + "_mask.png"
    )

    Image.fromarray(mask).save(mask_path)

    # Probability heatmap
    heatmap = (probability * 255).clip(0, 255).astype(np.uint8)

    heatmap_path = output_prefix.with_name(
        output_prefix.name + "_probability.png"
    )

    Image.fromarray(heatmap).save(heatmap_path)

    # Save raw probability
    npy_path = output_prefix.with_name(
        output_prefix.name + "_probability.npy"
    )

    np.save(npy_path, probability)

    # Statistics
    changed_pixels = int((probability >= 0.5).sum())
    total_pixels = probability.size
    change_percentage = 100.0 * changed_pixels / total_pixels

    print()
    print("=" * 60)
    print("CHANGE DETECTION RESULTS")
    print("=" * 60)
    print(f"Changed pixels : {changed_pixels:,}")
    print(f"Total pixels   : {total_pixels:,}")
    print(f"Change area    : {change_percentage:.2f}%")
    print(f"Minimum prob.  : {probability.min():.4f}")
    print(f"Maximum prob.  : {probability.max():.4f}")
    print(f"Mean prob.     : {probability.mean():.4f}")
    print()
    print("Saved:")
    print(mask_path)
    print(heatmap_path)
    print(npy_path)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="STTORM Sentinel-2 change detection"
    )

    parser.add_argument(
        "--before",
        required=True,
        help="Path to before-date .npy file"
    )

    parser.add_argument(
        "--after",
        required=True,
        help="Path to after-date .npy file"
    )

    parser.add_argument(
        "--out",
        default="outputs/change_prediction",
        help="Output file prefix"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STTORM CHANGE DETECTION")
    print("=" * 60)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    dev = device()

    print("Device:", dev)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------
    before_path = Path(args.before)
    after_path = Path(args.after)

    if not before_path.exists():
        raise FileNotFoundError(
            f"Before image not found:\n{before_path}"
        )

    if not after_path.exists():
        raise FileNotFoundError(
            f"After image not found:\n{after_path}"
        )

    print()
    print("Before:", before_path)
    print("After :", after_path)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    print()
    print("Loading trained model...")

    model = ChangeDetector().to(dev)

    checkpoint_path = OUTPUT_DIR / "change_best.pt"

    load_checkpoint(
        model,
        checkpoint_path,
        map_location=dev
    )

    model.eval()

    print("Model loaded successfully.")
    print("Checkpoint:", checkpoint_path)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    print()
    print("Running change detection...")

    probability = predict(
        model,
        before_path,
        after_path,
        dev
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    save_outputs(
        probability,
        args.out
    )

    # -----------------------------------------------------
    # GPU information
    # -----------------------------------------------------
    if torch.cuda.is_available():
        print()
        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated() / 1024**3,
                2
            ),
            "GB"
        )

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
