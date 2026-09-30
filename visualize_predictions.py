# visualize_predictions.py
# GeoEarthFM - Visualize Land Cover Predictions

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import random_split

from geoearthfm.config import *
from geoearthfm.datasets import EuroSATDataset
from geoearthfm.models import LandCoverModel
from geoearthfm.utils import seed_everything, device


def main():

    # -----------------------------------------------------
    # Setup
    # -----------------------------------------------------

    seed_everything(SEED)

    dev = device()

    print("Device:", dev)


    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    ds = EuroSATDataset(
        EUROSAT_DIR,
        train=True,
        image_size=IMAGE_SIZE
    )

    print("Classes:")
    print(ds.classes)


    # -----------------------------------------------------
    # Recreate SAME validation split
    # -----------------------------------------------------

    n_val = int(0.15 * len(ds))
    n_train = len(ds) - n_val

    train_ds, val_split = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )


    # -----------------------------------------------------
    # Validation dataset without augmentation
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
    # Load model
    # -----------------------------------------------------

    model = LandCoverModel(
        num_classes=len(ds.classes)
    ).to(dev)


    checkpoint_path = OUTPUT_DIR / "landcover_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=dev
    )


    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print("\nModel loaded successfully.")


    # -----------------------------------------------------
    # Find correct and incorrect predictions
    # -----------------------------------------------------

    correct_samples = []
    wrong_samples = []


    print("Finding predictions...")


    with torch.no_grad():

        for index in range(len(val_ds)):

            image, label = val_ds[index]

            x = image.unsqueeze(0).to(dev)

            logits, _ = model(x)

            prediction = logits.argmax(1).item()


            if prediction == label:

                if len(correct_samples) < 12:

                    correct_samples.append(
                        (image, label, prediction)
                    )

            else:

                if len(wrong_samples) < 12:

                    wrong_samples.append(
                        (image, label, prediction)
                    )


            # Stop once enough examples are collected
            if (
                len(correct_samples) >= 12
                and len(wrong_samples) >= 12
            ):
                break


    # -----------------------------------------------------
    # Function to display images
    # -----------------------------------------------------

    def display_samples(
        samples,
        title,
        filename
    ):

        rows = 3
        cols = 4

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(14, 11)
        )

        fig.suptitle(
            title,
            fontsize=18
        )


        for ax, sample in zip(
            axes.flat,
            samples
        ):

            image, actual, predicted = sample

            # Convert CHW -> HWC
            image_np = image.permute(
                1, 2, 0
            ).cpu().numpy()


            # Normalize for display
            image_np = image_np - image_np.min()

            if image_np.max() > 0:

                image_np = (
                    image_np / image_np.max()
                )


            ax.imshow(image_np)

            ax.set_title(
                f"Actual: {ds.classes[actual]}\n"
                f"Predicted: {ds.classes[predicted]}",
                fontsize=10
            )

            ax.axis("off")


        # Hide unused axes
        for ax in axes.flat[len(samples):]:

            ax.axis("off")


        plt.tight_layout()

        output_path = OUTPUT_DIR / filename

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


    # -----------------------------------------------------
    # Save correct predictions
    # -----------------------------------------------------

    display_samples(
        correct_samples,
        "Correct Land Cover Predictions",
        "correct_predictions.png"
    )


    # -----------------------------------------------------
    # Save wrong predictions
    # -----------------------------------------------------

    display_samples(
        wrong_samples,
        "Incorrect Land Cover Predictions",
        "wrong_predictions.png"
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n==============================================")
    print("       PREDICTION VISUALIZATION COMPLETE")
    print("==============================================")

    print(
        f"Correct examples saved: "
        f"{len(correct_samples)}"
    )

    print(
        f"Wrong examples saved: "
        f"{len(wrong_samples)}"
    )

    print("\nFiles:")

    print(
        OUTPUT_DIR / "correct_predictions.png"
    )

    print(
        OUTPUT_DIR / "wrong_predictions.png"
    )


if __name__ == "__main__":
    main()