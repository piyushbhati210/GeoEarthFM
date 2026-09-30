import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geoearthfm.config import OUTPUT_DIR
from geoearthfm.datasets import EuroSATDataset


# Load confusion matrix
cm_path = OUTPUT_DIR / "confusion_matrix.npy"
cm = np.load(cm_path)


# Get class names
ds = EuroSATDataset(
    # Use the same dataset location from config
    # The actual path comes from your project config
    __import__("geoearthfm.config", fromlist=["EUROSAT_DIR"]).EUROSAT_DIR,
    train=False
)

classes = ds.classes


# Create figure
plt.figure(figsize=(12, 10))

plt.imshow(cm, interpolation="nearest")

plt.title("EuroSAT Land Cover Confusion Matrix")
plt.colorbar()

plt.xticks(
    np.arange(len(classes)),
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(classes)),
    classes
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")


# Write values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()


# Save
output_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix image saved:")
print(output_path)