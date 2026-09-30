from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("outputs/all_test_pairs")

CSV_FILE = RESULTS_DIR / "all_test_pair_results.csv"

PLOTS_DIR = RESULTS_DIR / "final_plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load results
# ============================================================

print("=" * 70)
print("STTORM FINAL TEST RESULTS VISUALIZATION")
print("=" * 70)

print()
print("Loading:")
print(CSV_FILE)

df = pd.read_csv(CSV_FILE)

print()
print("Rows:", len(df))

print()
print("Columns:")
print(list(df.columns))


# ============================================================
# Keep only successful evaluations
# ============================================================

successful = df[
    df["Status"] == "Success"
].copy()

print()
print("Successful evaluations:", len(successful))


if len(successful) == 0:
    raise RuntimeError(
        "No successful evaluations found."
    )


# ============================================================
# Metrics
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Dice",
    "IoU",
]


# ============================================================
# 1. Overall metric plot
# ============================================================

overall = successful[metrics].mean()

plt.figure(figsize=(10, 6))

plt.bar(
    metrics,
    overall.values
)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title("Overall STTORM Test Performance")

for i, value in enumerate(overall.values):

    plt.text(
        i,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

overall_plot = (
    PLOTS_DIR /
    "overall_metrics.png"
)

plt.savefig(
    overall_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:")
print(overall_plot)


# ============================================================
# 2. Region-wise performance
# ============================================================

region_summary = (
    successful
    .groupby("Region")[metrics]
    .mean()
)

print()
print("=" * 70)
print("REGION-WISE PERFORMANCE")
print("=" * 70)

print(
    region_summary.to_string()
)


# ============================================================
# Region-wise grouped plot
# ============================================================

ax = region_summary.plot(
    kind="bar",
    figsize=(12, 7)
)

ax.set_title(
    "Region-wise STTORM Test Performance"
)

ax.set_ylabel("Score")

ax.set_xlabel("Region")

ax.set_ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Metric"
)

plt.tight_layout()

region_plot = (
    PLOTS_DIR /
    "region_wise_metrics.png"
)

plt.savefig(
    region_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:")
print(region_plot)


# ============================================================
# 3. Dice comparison by temporal pair
# ============================================================

successful["Pair"] = (
    successful["Region"]
    + ": "
    + successful["Before"]
    + " → "
    + successful["After"]
)

dice_sorted = successful.sort_values(
    "Dice",
    ascending=False
)

plt.figure(figsize=(14, 8))

plt.barh(
    dice_sorted["Pair"],
    dice_sorted["Dice"]
)

plt.xlim(0, 1)

plt.xlabel("Dice / F1")

plt.ylabel("Temporal Pair")

plt.title(
    "Dice Score Across All Temporal Pairs"
)

plt.tight_layout()

dice_plot = (
    PLOTS_DIR /
    "temporal_pair_dice.png"
)

plt.savefig(
    dice_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:")
print(dice_plot)


# ============================================================
# 4. IoU comparison by temporal pair
# ============================================================

iou_sorted = successful.sort_values(
    "IoU",
    ascending=False
)

plt.figure(figsize=(14, 8))

plt.barh(
    iou_sorted["Pair"],
    iou_sorted["IoU"]
)

plt.xlim(0, 1)

plt.xlabel("IoU")

plt.ylabel("Temporal Pair")

plt.title(
    "IoU Across All Temporal Pairs"
)

plt.tight_layout()

iou_plot = (
    PLOTS_DIR /
    "temporal_pair_iou.png"
)

plt.savefig(
    iou_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:")
print(iou_plot)


# ============================================================
# 5. Precision / Recall comparison
# ============================================================

region_pr = (
    successful
    .groupby("Region")[
        ["Precision", "Recall"]
    ]
    .mean()
)

ax = region_pr.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Region-wise Precision and Recall"
)

ax.set_ylabel("Score")

ax.set_xlabel("Region")

ax.set_ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.tight_layout()

pr_plot = (
    PLOTS_DIR /
    "region_precision_recall.png"
)

plt.savefig(
    pr_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:")
print(pr_plot)


# ============================================================
# 6. Best and lowest temporal pairs
# ============================================================

best_pairs = successful.sort_values(
    "Dice",
    ascending=False
).head(5)

lowest_pairs = successful.sort_values(
    "Dice",
    ascending=True
).head(5)


# ============================================================
# Save best / lowest pairs
# ============================================================

best_pairs[
    [
        "Region",
        "Before",
        "After",
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]
].to_csv(
    RESULTS_DIR / "top_5_temporal_pairs.csv",
    index=False
)

lowest_pairs[
    [
        "Region",
        "Before",
        "After",
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]
].to_csv(
    RESULTS_DIR / "lowest_5_temporal_pairs.csv",
    index=False
)


# ============================================================
# 7. Final summary CSV
# ============================================================

final_summary = pd.DataFrame(
    {
        "Metric": metrics,
        "Overall Score": [
            overall[m]
            for m in metrics
        ],
    }
)

final_summary.to_csv(
    RESULTS_DIR / "final_overall_summary.csv",
    index=False
)


# ============================================================
# Final console output
# ============================================================

print()
print("=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

for metric in metrics:

    print(
        f"{metric:<12}: "
        f"{overall[metric]:.4f}"
    )


print()
print("=" * 70)
print("TOP 5 TEMPORAL PAIRS BY DICE")
print("=" * 70)

print(
    best_pairs[
        [
            "Region",
            "Before",
            "After",
            "Dice",
            "IoU",
        ]
    ].to_string(index=False)
)


print()
print("=" * 70)
print("LOWEST 5 TEMPORAL PAIRS BY DICE")
print("=" * 70)

print(
    lowest_pairs[
        [
            "Region",
            "Before",
            "After",
            "Dice",
            "IoU",
        ]
    ].to_string(index=False)
)


print()
print("=" * 70)
print("FINAL VISUALIZATION COMPLETE")
print("=" * 70)

print()
print("Plots saved in:")
print(PLOTS_DIR)

print()
print("Additional result files:")
print(
    RESULTS_DIR /
    "top_5_temporal_pairs.csv"
)

print(
    RESULTS_DIR /
    "lowest_5_temporal_pairs.csv"
)

print(
    RESULTS_DIR /
    "final_overall_summary.csv"
)