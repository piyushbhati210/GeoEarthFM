from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "outputs/all_test_pairs/all_test_pair_results.csv"
)

OUTPUT_DIR = Path(
    "outputs/all_test_pairs/final_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("STTORM FINAL RESULTS ANALYSIS")
    print("=" * 70)

    print()
    print("Loading:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print()
    print("Rows:", len(df))
    print()
    print("Columns:")
    print(df.columns.tolist())

    # Keep only successful evaluations
    if "Status" in df.columns:
        df = df[
            df["Status"].astype(str).str.lower()
            == "success"
        ].copy()

    print()
    print(
        "Successful evaluations:",
        len(df)
    )

    # Convert dates
    df["Before"] = pd.to_datetime(
        df["Before"],
        errors="coerce"
    )

    df["After"] = pd.to_datetime(
        df["After"],
        errors="coerce"
    )

    # Calculate temporal interval
    df["Days_Between"] = (
        df["After"] - df["Before"]
    ).dt.days

    # Numeric columns
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]

    for column in metric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

def calculate_overall(df):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]

    overall = df[metrics].mean()

    print()
    print("=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy  : {overall['Accuracy']:.4f}"
    )

    print(
        f"Precision : {overall['Precision']:.4f}"
    )

    print(
        f"Recall    : {overall['Recall']:.4f}"
    )

    print(
        f"Dice      : {overall['Dice']:.4f}"
    )

    print(
        f"IoU       : {overall['IoU']:.4f}"
    )

    output = pd.DataFrame({
        "Metric": metrics,
        "Value": [
            overall["Accuracy"],
            overall["Precision"],
            overall["Recall"],
            overall["Dice"],
            overall["IoU"],
        ],
    })

    output_path = Path(
        "outputs/all_test_pairs/"
        "final_overall_summary.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print("Saved:")
    print(output_path)

    return overall


# ============================================================
# REGION-WISE PERFORMANCE
# ============================================================

def calculate_region_performance(df):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]

    region_summary = (
        df.groupby("Region")[metrics]
        .mean()
        .round(4)
    )

    print()
    print("=" * 70)
    print("REGION-WISE PERFORMANCE")
    print("=" * 70)

    print(region_summary)

    output_path = Path(
        "outputs/all_test_pairs/"
        "final_region_summary.csv"
    )

    region_summary.to_csv(
        output_path
    )

    print()
    print("Saved:")
    print(output_path)

    return region_summary


# ============================================================
# TEMPORAL INTERVAL PERFORMANCE
# ============================================================

def calculate_temporal_performance(df):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]

    temporal_summary = (
        df.groupby("Days_Between")[metrics]
        .mean()
        .sort_index()
        .round(4)
    )

    print()
    print("=" * 70)
    print("TEMPORAL INTERVAL PERFORMANCE")
    print("=" * 70)

    print(temporal_summary)

    output_path = Path(
        "outputs/all_test_pairs/"
        "temporal_interval_summary.csv"
    )

    temporal_summary.to_csv(
        output_path
    )

    print()
    print("Saved:")
    print(output_path)

    return temporal_summary


# ============================================================
# TOP 5 TEMPORAL PAIRS
# ============================================================

def save_top_pairs(df):

    columns = [
        "Region",
        "Before",
        "After",
        "Dice",
        "IoU",
    ]

    top5 = (
        df.sort_values(
            by="Dice",
            ascending=False
        )
        .head(5)
        .copy()
    )

    top5["Before"] = top5["Before"].dt.strftime(
        "%Y-%m-%d"
    )

    top5["After"] = top5["After"].dt.strftime(
        "%Y-%m-%d"
    )

    print()
    print("=" * 70)
    print("TOP 5 TEMPORAL PAIRS BY DICE")
    print("=" * 70)

    print(
        top5[columns].to_string(
            index=False
        )
    )

    output_path = Path(
        "outputs/all_test_pairs/"
        "final_top_5_temporal_pairs.csv"
    )

    top5[columns].to_csv(
        output_path,
        index=False
    )

    print()
    print("Saved:")
    print(output_path)

    return top5


# ============================================================
# LOWEST 5 TEMPORAL PAIRS
# ============================================================

def save_lowest_pairs(df):

    columns = [
        "Region",
        "Before",
        "After",
        "Dice",
        "IoU",
    ]

    lowest5 = (
        df.sort_values(
            by="Dice",
            ascending=True
        )
        .head(5)
        .copy()
    )

    lowest5["Before"] = lowest5["Before"].dt.strftime(
        "%Y-%m-%d"
    )

    lowest5["After"] = lowest5["After"].dt.strftime(
        "%Y-%m-%d"
    )

    print()
    print("=" * 70)
    print("LOWEST 5 TEMPORAL PAIRS BY DICE")
    print("=" * 70)

    print(
        lowest5[columns].to_string(
            index=False
        )
    )

    output_path = Path(
        "outputs/all_test_pairs/"
        "final_lowest_5_temporal_pairs.csv"
    )

    lowest5[columns].to_csv(
        output_path,
        index=False
    )

    print()
    print("Saved:")
    print(output_path)

    return lowest5


# ============================================================
# PLOT 1: OVERALL PERFORMANCE
# ============================================================

def plot_overall_performance(overall):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]

    values = [
        overall["Accuracy"],
        overall["Precision"],
        overall["Recall"],
        overall["Dice"],
        overall["IoU"],
    ]

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        metrics,
        values
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "STTORM Overall Test Performance"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        values
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "overall_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 2: REGION PERFORMANCE
# ============================================================

def plot_region_performance(region_summary):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Dice",
        "IoU",
    ]

    ax = region_summary[
        metrics
    ].plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_title(
        "STTORM Region-Wise Performance"
    )

    ax.set_xlabel(
        "Region"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "region_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 3: DICE VS TEMPORAL INTERVAL
# ============================================================

def plot_dice_vs_temporal(
    temporal_summary
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        temporal_summary.index,
        temporal_summary["Dice"],
        marker="o"
    )

    plt.xlabel(
        "Temporal Interval (Days)"
    )

    plt.ylabel(
        "Dice / F1"
    )

    plt.title(
        "Dice Performance vs Temporal Interval"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "dice_vs_temporal_interval.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 4: IoU VS TEMPORAL INTERVAL
# ============================================================

def plot_iou_vs_temporal(
    temporal_summary
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        temporal_summary.index,
        temporal_summary["IoU"],
        marker="o"
    )

    plt.xlabel(
        "Temporal Interval (Days)"
    )

    plt.ylabel(
        "IoU"
    )

    plt.title(
        "IoU Performance vs Temporal Interval"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "iou_vs_temporal_interval.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 5: PRECISION VS RECALL BY REGION
# ============================================================

def plot_region_precision_recall(
    region_summary
):

    regions = region_summary.index

    x = np.arange(
        len(regions)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x - width / 2,
        region_summary["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x + width / 2,
        region_summary["Recall"],
        width,
        label="Recall"
    )

    plt.xticks(
        x,
        regions
    )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Region"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Precision vs Recall by Region"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "region_precision_recall.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 6: DICE DISTRIBUTION
# ============================================================

def plot_dice_distribution(df):

    regions = sorted(
        df["Region"].unique()
    )

    data = []

    for region in regions:

        values = (
            df.loc[
                df["Region"] == region,
                "Dice"
            ]
            .dropna()
            .values
        )

        data.append(values)

    plt.figure(
        figsize=(10, 7)
    )

    # IMPORTANT:
    # New Matplotlib versions use tick_labels
    # instead of labels.
    plt.boxplot(
        data,
        tick_labels=regions,
        patch_artist=True
    )

    plt.xlabel(
        "Region"
    )

    plt.ylabel(
        "Dice / F1"
    )

    plt.title(
        "Dice Score Distribution by Region"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "dice_distribution_by_region.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 7: IoU DISTRIBUTION
# ============================================================

def plot_iou_distribution(df):

    regions = sorted(
        df["Region"].unique()
    )

    data = []

    for region in regions:

        values = (
            df.loc[
                df["Region"] == region,
                "IoU"
            ]
            .dropna()
            .values
        )

        data.append(values)

    plt.figure(
        figsize=(10, 7)
    )

    plt.boxplot(
        data,
        tick_labels=regions,
        patch_artist=True
    )

    plt.xlabel(
        "Region"
    )

    plt.ylabel(
        "IoU"
    )

    plt.title(
        "IoU Distribution by Region"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "iou_distribution_by_region.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# PLOT 8: ALL TEMPORAL PAIRS DICE
# ============================================================

def plot_all_pair_dice(df):

    plot_df = df.copy()

    plot_df["Pair"] = (
        plot_df["Region"]
        + "\n"
        + plot_df["Before"].dt.strftime(
            "%m-%d"
        )
        + " → "
        + plot_df["After"].dt.strftime(
            "%m-%d"
        )
    )

    plot_df = plot_df.sort_values(
        "Dice",
        ascending=False
    )

    plt.figure(
        figsize=(15, 8)
    )

    plt.bar(
        range(len(plot_df)),
        plot_df["Dice"]
    )

    plt.xlabel(
        "Temporal Pair"
    )

    plt.ylabel(
        "Dice / F1"
    )

    plt.title(
        "Dice Score for All 40 Test Temporal Pairs"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        range(len(plot_df)),
        plot_df["Pair"],
        rotation=90,
        fontsize=7
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "all_temporal_pairs_dice.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# SAVE COMPLETE ANALYSIS CSV
# ============================================================

def save_complete_analysis(df):

    output = df.copy()

    output["Before"] = (
        output["Before"]
        .dt.strftime("%Y-%m-%d")
    )

    output["After"] = (
        output["After"]
        .dt.strftime("%Y-%m-%d")
    )

    output_path = (
        Path("outputs/all_test_pairs")
        / "complete_final_analysis.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print("Saved:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    if len(df) == 0:
        raise RuntimeError(
            "No successful evaluations found."
        )

    overall = calculate_overall(
        df
    )

    region_summary = (
        calculate_region_performance(
            df
        )
    )

    temporal_summary = (
        calculate_temporal_performance(
            df
        )
    )

    save_top_pairs(
        df
    )

    save_lowest_pairs(
        df
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_overall_performance(
        overall
    )

    plot_region_performance(
        region_summary
    )

    plot_dice_vs_temporal(
        temporal_summary
    )

    plot_iou_vs_temporal(
        temporal_summary
    )

    plot_region_precision_recall(
        region_summary
    )

    plot_dice_distribution(
        df
    )

    plot_iou_distribution(
        df
    )

    plot_all_pair_dice(
        df
    )

    save_complete_analysis(
        df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("All analysis files and plots saved in:")

    print(
        OUTPUT_DIR
    )

    print()
    print("Main output files:")

    print(
        "1. overall_performance.png"
    )

    print(
        "2. region_performance.png"
    )

    print(
        "3. dice_vs_temporal_interval.png"
    )

    print(
        "4. iou_vs_temporal_interval.png"
    )

    print(
        "5. region_precision_recall.png"
    )

    print(
        "6. dice_distribution_by_region.png"
    )

    print(
        "7. iou_distribution_by_region.png"
    )

    print(
        "8. all_temporal_pairs_dice.png"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()