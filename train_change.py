import argparse

import torch

from torch import nn, optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from geoearthfm.config import *

from geoearthfm.datasets import STTORMChangeDataset

from geoearthfm.models import ChangeDetector

from geoearthfm.utils import (
    seed_everything,
    device,
    save_checkpoint
)


# ============================================================
# Dice Score
# ============================================================

def dice_score(
    logits,
    target,
    eps=1e-6
):

    pred = (
        torch.sigmoid(logits) > 0.5
    ).float()

    intersection = (
        pred * target
    ).sum()

    return (
        2 * intersection + eps
    ) / (
        pred.sum()
        + target.sum()
        + eps
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_CHANGE
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_CHANGE
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000
    )

    args = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    seed_everything(
        SEED
    )

    # ========================================================
    # Device
    # ========================================================

    dev = device()

    print("=" * 60)

    print(
        "CHANGE DETECTION TRAINING"
    )

    print("=" * 60)

    print(
        "Device:",
        dev
    )

    # ========================================================
    # GPU information
    # ========================================================

    if torch.cuda.is_available():

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        gpu_properties = (
            torch.cuda.get_device_properties(0)
        )

        total_vram = (
            gpu_properties.total_memory
            / 1024**3
        )

        print(
            "GPU:",
            gpu_name
        )

        print(
            "VRAM:",
            round(total_vram, 2),
            "GB"
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    else:

        print(
            "WARNING: CUDA is not available!"
        )

    # ========================================================
    # Training configuration
    # ========================================================

    print(
        "Mixed Precision:",
        "Enabled"
        if torch.cuda.is_available()
        else "Disabled"
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Workers:",
        NUM_WORKERS
    )

    print(
        "Maximum samples:",
        args.max_samples
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\nLoading STTORM dataset..."
    )

    ds = STTORMChangeDataset(

        STTORM_DIR,

        split="train",

        tile_size=CHANGE_TILE_SIZE,

        stride=CHANGE_STRIDE,

        max_samples=args.max_samples,

        cache=True
    )

    print(
        "Total samples:",
        len(ds)
    )

    # ========================================================
    # Train / Validation split
    # ========================================================

    n_val = max(
        1,
        int(0.15 * len(ds))
    )

    n_train = (
        len(ds) - n_val
    )

    train_ds, val_ds = (
        torch.utils.data.random_split(

            ds,

            [
                n_train,
                n_val
            ],

            generator=torch.Generator()
                .manual_seed(SEED)
        )
    )

    print(
        "Training samples:",
        len(train_ds)
    )

    print(
        "Validation samples:",
        len(val_ds)
    )

    # ========================================================
    # DataLoader settings
    # ========================================================

    loader_kwargs = {

        "batch_size":
            args.batch_size,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            torch.cuda.is_available(),

    }

    # --------------------------------------------------------
    # Windows multiprocessing
    # --------------------------------------------------------

    if NUM_WORKERS > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    # ========================================================
    # Train loader
    # ========================================================

    train_loader = DataLoader(

        train_ds,

        shuffle=True,

        drop_last=True,

        **loader_kwargs

    )

    # ========================================================
    # Validation loader
    # ========================================================

    val_loader = DataLoader(

        val_ds,

        shuffle=False,

        **loader_kwargs

    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nCreating model..."
    )

    model = ChangeDetector().to(
        dev
    )

    print(
        "Model moved to:",
        next(
            model.parameters()
        ).device
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LR_CHANGE,

        weight_decay=1e-4

    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    # ========================================================
    # Mixed Precision
    # ========================================================

    use_amp = (
        torch.cuda.is_available()
    )

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=use_amp

    )

    # ========================================================
    # Best score
    # ========================================================

    best = 0.0

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Train mode
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        progress = tqdm(

            train_loader,

            desc=(
                f"Epoch "
                f"{epoch}/{args.epochs} train"
            )

        )

        # ----------------------------------------------------
        # Batch loop
        # ----------------------------------------------------

        for (
            before,
            after,
            mask
        ) in progress:

            # ------------------------------------------------
            # GPU transfer
            # ------------------------------------------------

            before = before.to(

                dev,

                non_blocking=True

            )

            after = after.to(

                dev,

                non_blocking=True

            )

            mask = mask.to(

                dev,

                non_blocking=True

            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16,

                enabled=use_amp

            ):

                logits = model(

                    before,

                    after

                )

                loss = criterion(

                    logits,

                    mask

                )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            train_loss += (

                loss.item()
                * before.size(0)

            )

            # ------------------------------------------------
            # Progress bar
            # ------------------------------------------------

            progress.set_postfix(

                loss=(
                    f"{loss.item():.4f}"
                )

            )

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        dices = []

        with torch.no_grad():

            for (
                before,
                after,
                mask
            ) in val_loader:

                # ------------------------------------------------
                # GPU
                # ------------------------------------------------

                before = before.to(

                    dev,

                    non_blocking=True

                )

                after = after.to(

                    dev,

                    non_blocking=True

                )

                mask = mask.to(

                    dev,

                    non_blocking=True

                )

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                with torch.autocast(

                    device_type="cuda",

                    dtype=torch.float16,

                    enabled=use_amp

                ):

                    logits = model(

                        before,

                        after

                    )

                # ------------------------------------------------
                # Dice
                # ------------------------------------------------

                dices.append(

                    dice_score(

                        logits,

                        mask

                    ).item()

                )

        # ====================================================
        # Metrics
        # ====================================================

        score = (
            sum(dices)
            / max(1, len(dices))
        )

        average_loss = (

            train_loss
            / max(1, len(train_ds))

        )

        print(
            f"loss={average_loss:.4f} "
            f"val_dice={score:.4f}"
        )

        # ====================================================
        # Save best model
        # ====================================================

        if score > best:

            best = score

            save_checkpoint(

                model,

                OUTPUT_DIR
                / "change_best.pt",

                val_dice=best

            )

            print(
                f"Best model saved: "
                f"Dice={best:.4f}"
            )

        # ====================================================
        # GPU memory information
        # ====================================================

        if torch.cuda.is_available():

            allocated = (

                torch.cuda.memory_allocated()
                / 1024**3

            )

            reserved = (

                torch.cuda.memory_reserved()
                / 1024**3

            )

            print(

                "GPU memory: "

                f"{allocated:.2f} GB allocated / "

                f"{reserved:.2f} GB reserved"

            )

    # ========================================================
    # Training complete
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "Best validation Dice:",
        best
    )

    print(
        "Saved:",
        OUTPUT_DIR
        / "change_best.pt"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()