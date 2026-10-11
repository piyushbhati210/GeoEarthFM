import numpy as np
from pathlib import Path

folder = Path("dataset/test/brazil/all_bands")

for file in sorted(folder.glob("*.npy")):
    data = np.load(file)

    print("\nFile:", file.name)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum:", data.min())
    print("Maximum:", data.max())