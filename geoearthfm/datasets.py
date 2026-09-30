from pathlib import Path
import datetime as dt

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

from .config import STTORM_BAND_INDICES

from .utils import normalize_spectral, to_chw


# ============================================================
# EuroSAT Dataset
# ============================================================

class EuroSATDataset(Dataset):

    def __init__(
        self,
        root,
        train=True,
        image_size=224
    ):

        self.root = Path(root)

        self.transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.RandomHorizontalFlip()
            if train
            else transforms.Lambda(
                lambda x: x
            ),

            transforms.RandomVerticalFlip()
            if train
            else transforms.Lambda(
                lambda x: x
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        # ----------------------------------------------------
        # Classes
        # ----------------------------------------------------

        classes = sorted([
            p.name
            for p in self.root.iterdir()
            if p.is_dir()
        ])

        self.classes = classes

        self.class_to_idx = {
            c: i
            for i, c in enumerate(classes)
        }

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        self.samples = []

        for c in classes:

            class_dir = self.root / c

            for p in sorted(
                class_dir.glob("*")
            ):

                if p.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tif",
                    ".tiff"
                }:

                    self.samples.append(
                        (
                            p,
                            self.class_to_idx[c]
                        )
                    )

        if not self.samples:

            raise RuntimeError(
                f"No EuroSAT images found in {self.root}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# STTORM Change Detection Dataset
# ============================================================

class STTORMChangeDataset(Dataset):

    """
    Optimized STTORM change-detection dataset.

    Structure:

        dataset/
            train/
                event/
                    sequence/
                        all_bands/
                            date1.npy
                            date2.npy
                            ...
                        mask.npy

    The latest image is treated as AFTER.

    Earlier images are treated as BEFORE.

    Optimization:
        1. Cache loaded NPY files.
        2. Crop before normalization.
        3. Select only required bands.
        4. Convert arrays to contiguous float32 arrays.
    """

    def __init__(
        self,
        root,
        split="train",
        tile_size=128,
        stride=128,
        max_samples=None,
        cache=True
    ):

        self.root = Path(root) / split

        self.tile_size = tile_size

        self.stride = stride

        self.cache_enabled = cache

        # ----------------------------------------------------
        # RAM cache
        # ----------------------------------------------------

        self.image_cache = {}

        self.mask_cache = {}

        # ----------------------------------------------------
        # Sample index
        # ----------------------------------------------------

        self.samples = []

        self._build_index()

        # ----------------------------------------------------
        # Limit samples
        # ----------------------------------------------------

        if max_samples is not None:

            self.samples = self.samples[
                :max_samples
            ]

        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        if not self.samples:

            raise RuntimeError(
                f"No STTORM samples found under {self.root}. "
                "Expected event/sequence/all_bands/*.npy "
                "and mask.npy."
            )

    # ========================================================
    # Date parser
    # ========================================================

    def _date_key(self, p):

        try:

            return dt.date.fromisoformat(
                p.stem
            )

        except Exception:

            return dt.date.min

    # ========================================================
    # Build sample index
    # ========================================================

    def _build_index(self):

        event_dirs = sorted([

            p
            for p in self.root.iterdir()
            if p.is_dir()

        ])

        for event_dir in event_dirs:

            sequence_dirs = [

                p
                for p in event_dir.iterdir()
                if p.is_dir()

            ]

            # ------------------------------------------------
            # Direct event structure
            # ------------------------------------------------

            if not sequence_dirs:

                sequence_dirs = [
                    event_dir
                ]

            # ------------------------------------------------
            # Process sequence
            # ------------------------------------------------

            for seq in sequence_dirs:

                img_dir = (
                    seq / "all_bands"
                )

                mask_path = (
                    seq / "mask.npy"
                )

                if (
                    not img_dir.exists()
                    or not mask_path.exists()
                ):

                    continue

                # ------------------------------------------------
                # Find temporal images
                # ------------------------------------------------

                images = sorted(

                    [
                        p
                        for p in img_dir.glob(
                            "*.npy"
                        )

                        if p.name != "mask.npy"
                    ],

                    key=self._date_key
                )

                if len(images) < 2:

                    continue

                # ------------------------------------------------
                # Read mask dimensions
                # ------------------------------------------------

                mask = np.load(
                    mask_path,
                    mmap_mode="r"
                )

                if mask.ndim > 2:

                    mask = np.squeeze(
                        mask
                    )

                h, w = mask.shape[:2]

                # ------------------------------------------------
                # Latest image = AFTER
                # ------------------------------------------------

                after = images[-1]

                # ------------------------------------------------
                # Earlier images = BEFORE
                # ------------------------------------------------

                before_images = images[:-1]

                # ------------------------------------------------
                # Generate tiles
                # ------------------------------------------------

                for before in before_images:

                    for y in range(
                        0,
                        h - self.tile_size + 1,
                        self.stride
                    ):

                        for x in range(
                            0,
                            w - self.tile_size + 1,
                            self.stride
                        ):

                            self.samples.append(

                                (
                                    before,
                                    after,
                                    mask_path,
                                    y,
                                    x
                                )

                            )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # Load image with RAM cache
    # ========================================================

    def _load_npy(self, path):

        path = str(path)

        # ----------------------------------------------------
        # RAM cache
        # ----------------------------------------------------

        if self.cache_enabled:

            if path in self.image_cache:

                return self.image_cache[path]

        # ----------------------------------------------------
        # Load NPY
        # ----------------------------------------------------

        array = np.load(
            path
        )

        # ----------------------------------------------------
        # Float32
        # ----------------------------------------------------

        if array.dtype != np.float32:

            array = array.astype(
                np.float32,
                copy=False
            )

        # ----------------------------------------------------
        # Store in cache
        # ----------------------------------------------------

        if self.cache_enabled:

            self.image_cache[path] = array

        return array

    # ========================================================
    # Load mask with cache
    # ========================================================

    def _load_mask(self, path):

        path = str(path)

        if path in self.mask_cache:

            return self.mask_cache[path]

        mask = np.load(
            path
        )

        if mask.ndim > 2:

            mask = np.squeeze(
                mask
            )

        mask = mask.astype(
            np.float32,
            copy=False
        )

        self.mask_cache[path] = mask

        return mask

    # ========================================================
    # Get sample
    # ========================================================

    def __getitem__(self, idx):

        (
            before_path,
            after_path,
            mask_path,
            y,
            x
        ) = self.samples[idx]

        # ----------------------------------------------------
        # Load arrays
        # ----------------------------------------------------

        before = self._load_npy(
            before_path
        )

        after = self._load_npy(
            after_path
        )

        mask = self._load_mask(
            mask_path
        )

        # ----------------------------------------------------
        # Tile coordinates
        # ----------------------------------------------------

        y2 = y + self.tile_size

        x2 = x + self.tile_size

        # ----------------------------------------------------
        # CROP FIRST
        # ----------------------------------------------------

        before = before[
            y:y2,
            x:x2
        ]

        after = after[
            y:y2,
            x:x2
        ]

        mask = mask[
            y:y2,
            x:x2
        ]

        # ----------------------------------------------------
        # Select 10 Sentinel-2 bands
        # ----------------------------------------------------

        before = before[
            :,
            :,
            STTORM_BAND_INDICES
        ]

        after = after[
            :,
            :,
            STTORM_BAND_INDICES
        ]

        # ----------------------------------------------------
        # Normalize ONLY 128x128 tile
        # ----------------------------------------------------

        before = normalize_spectral(
            before
        )

        after = normalize_spectral(
            after
        )

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        before = to_chw(
            before
        )

        after = to_chw(
            after
        )

        # ----------------------------------------------------
        # Contiguous arrays
        # ----------------------------------------------------

        before = np.ascontiguousarray(
            before,
            dtype=np.float32
        )

        after = np.ascontiguousarray(
            after,
            dtype=np.float32
        )

        mask = np.ascontiguousarray(
            mask,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Return tensors
        # ----------------------------------------------------

        return (

            torch.from_numpy(
                before
            ),

            torch.from_numpy(
                after
            ),

            torch.from_numpy(
                mask
            ).unsqueeze(0)

        )