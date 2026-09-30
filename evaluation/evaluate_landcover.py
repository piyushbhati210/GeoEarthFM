# evaluate_landcover.py
# GeoEarthFM - Land Cover Model Evaluation

import torch
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import numpy as np

from geoearthfm.config import *
from geoearthfm.datasets import EuroSATDataset
from geoearthfm.models import LandCoverModel
from geoearthfm.utils import seed_everything, device


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # -----------------------------------------------------
    # Set seed and device
    # -----------------------------------------------------

    seed_everything(SEED)

    dev = device()

    print("Device:", dev)


    # -----------------------------------------------------
    # Load EuroSAT dataset
    # -----------------------------------------------------

    ds = EuroSATDataset(
        EUROSAT_DIR,
        train=True,
        image_size=IMAGE_SIZE
    )

    print("Number of classes:", len(ds.classes))
    print("Classes:", ds.classes)


    # -----------------------------------------------------
    # Create the SAME validation split used during training
    # -----------------------------------------------------

    n_val = int(0.15 * len(ds))
    n_train = len(ds) - n_val

    train_ds, val_split = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )


    # -----------------------------------------------------
    # Create validation dataset WITHOUT augmentation
    # -----------------------------------------------------

    val_base = EuroSATDataset(
        EUROSAT_DIR,
        train=False,
        image_size=IMAGE_SIZE
    )

    val_ds = torch.utils.data.Subset(
        val_base,
        val_split.indices
    )


    # -----------------------------------------------------
    # Validation DataLoader
    # -----------------------------------------------------

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_CLASSIFICATION,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_ds))


    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = LandCoverModel(
        num_classes=len(ds.classes)
    ).to(dev)


    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint_path = OUTPUT_DIR / "landcover_best.pt"

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=dev
    )


    # -----------------------------------------------------
    # IMPORTANT:
    # Your checkpoint contains:
    #
    # "model"
    # "classes"
    # "val_accuracy"
    #
    # Therefore load checkpoint["model"]
    # -----------------------------------------------------

    if isinstance(checkpoint, dict) and "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

        print("Checkpoint loaded successfully.")

        if "classes" in checkpoint:

            print(
                "Checkpoint classes:",
                checkpoint["classes"]
            )

        if "val_accuracy" in checkpoint:

            print(
                "Saved best validation accuracy: "
                f"{checkpoint['val_accuracy'] * 100:.2f}%"
            )

    else:

        print(
            "ERROR: Unexpected checkpoint format."
        )

        return


    # -----------------------------------------------------
    # Evaluation mode
    # -----------------------------------------------------

    model.eval()


    # -----------------------------------------------------
    # Store predictions and actual labels
    # -----------------------------------------------------

    all_predictions = []
    all_labels = []


    correct = 0
    total = 0


    # -----------------------------------------------------
    # Run model on validation dataset
    # -----------------------------------------------------

    print("\nEvaluating model...")

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(dev)
            y = y.to(dev)

            # Model prediction
            logits, _ = model(x)

            predictions = logits.argmax(dim=1)


            # Accuracy calculation
            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)


            # Store predictions
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Store actual labels
            all_labels.extend(
                y.cpu().numpy()
            )


    # -----------------------------------------------------
    # Convert to NumPy arrays
    # -----------------------------------------------------

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )


    # =====================================================
    # ACCURACY
    # =====================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    print("\n")
    print("==============================================")
    print("           LAND COVER EVALUATION")
    print("==============================================")


    print(
        f"\nValidation Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print("\n")
    print("==============================================")
    print("           CLASSIFICATION REPORT")
    print("==============================================")


    report = classification_report(
        all_labels,
        all_predictions,
        target_names=ds.classes,
        digits=4,
        zero_division=0
    )

    print(report)


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )


    print("\n")
    print("==============================================")
    print("              CONFUSION MATRIX")
    print("==============================================")


    print(cm)


    # =====================================================
    # CLASS-WISE ANALYSIS
    # =====================================================

    print("\n")
    print("==============================================")
    print("         CLASS-WISE CONFUSION")
    print("==============================================")


    for i, class_name in enumerate(ds.classes):

        # Copy row
        row = cm[i].copy()

        # Remove correct predictions
        row[i] = 0


        if row.sum() > 0:

            confused_index = np.argmax(row)

            confused_count = row[confused_index]

            print(
                f"{class_name:25s} "
                f"-> {ds.classes[confused_index]:25s} "
                f"({confused_count} images)"
            )

        else:

            print(
                f"{class_name:25s} "
                f"-> No major confusion"
            )


    # =====================================================
    # SAVE CONFUSION MATRIX
    # =====================================================

    output_file = (
        OUTPUT_DIR / "confusion_matrix.npy"
    )

    np.save(
        output_file,
        cm
    )


    print("\n")
    print("Confusion matrix saved:")
    print(output_file)


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n")
    print("==============================================")
    print("                 SUMMARY")
    print("==============================================")


    print(
        f"Validation samples : {len(all_labels)}"
    )

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Total predictions  : {total}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nEvaluation completed successfully."
    )


# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":
    main()