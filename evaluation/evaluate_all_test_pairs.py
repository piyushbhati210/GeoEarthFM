import subprocess
import sys
from pathlib import Path
import itertools
import re

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

TEST_DIR = Path("dataset/test")
OUTPUT_DIR = Path("outputs/all_test_pairs")

REGIONS = [
    "brazil",
    "germany",
    "laos",
    "niger",
]


# =========================================================
# Extract metrics from evaluate_change.py output
# =========================================================

def parse_metrics(text):

    patterns = {
        "Accuracy": r"Accuracy\s+:\s+([0-9.]+)",
        "Precision": r"Precision\s+:\s+([0-9.]+)",
        "Recall": r"Recall\s+:\s+([0-9.]+)",
        "Dice": r"Dice / F1\s+:\s+([0-9.]+)",
        "IoU": r"IoU\s+:\s+([0-9.]+)",
    }

    metrics = {}

    for name, pattern in patterns.items():

        match = re.search(pattern, text)

        if match:
            metrics[name] = float(match.group(1))
        else:
            metrics[name] = np.nan

    return metrics


# =========================================================
# Run Python script using CURRENT VIRTUAL ENVIRONMENT
# =========================================================

def run_python_script(script, arguments):

    command = [
        sys.executable,
        script,
    ] + arguments

    return subprocess.run(
        command,
        capture_output=True,
        text=True
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("STTORM FULL TEST-SET TEMPORAL EVALUATION")
    print("=" * 70)

    print()
    print("Python executable:")
    print(sys.executable)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    total_pairs = 0

    # =====================================================
    # Process each region
    # =====================================================

    for region in REGIONS:

        region_dir = TEST_DIR / region
        bands_dir = region_dir / "all_bands"
        ground_truth = region_dir / "mask.npy"

        # -------------------------------------------------
        # Check required paths
        # -------------------------------------------------

        if not region_dir.exists():

            print()
            print(f"WARNING: Region not found: {region_dir}")
            continue

        if not bands_dir.exists():

            print()
            print(f"WARNING: all_bands folder not found:")
            print(bands_dir)
            continue

        if not ground_truth.exists():

            print()
            print(f"WARNING: Ground truth not found:")
            print(ground_truth)
            continue

        # -------------------------------------------------
        # Find dates
        # -------------------------------------------------

        dates = sorted(
            bands_dir.glob("*.npy")
        )

        print()
        print("=" * 70)
        print("REGION:", region.upper())
        print("=" * 70)

        print("Dates found:")

        for date in dates:
            print(" ", date.name)

        # -------------------------------------------------
        # Generate all unique temporal pairs
        # -------------------------------------------------

        pairs = list(
            itertools.combinations(
                dates,
                2
            )
        )

        print()
        print("Temporal pairs:", len(pairs))

        # =================================================
        # Process every pair
        # =================================================

        for pair_number, (before, after) in enumerate(
            pairs,
            start=1
        ):

            total_pairs += 1

            before_date = before.stem
            after_date = after.stem

            pair_name = (
                f"{region}_{before_date}_{after_date}"
            )

            print()
            print("-" * 70)

            print(
                f"[{pair_number}/{len(pairs)}] "
                f"{region}: "
                f"{before_date} -> {after_date}"
            )

            print("-" * 70)

            # -------------------------------------------------
            # Output locations
            # -------------------------------------------------

            prediction_output = (
                OUTPUT_DIR / pair_name
            )

            prediction_file = (
                OUTPUT_DIR /
                f"{pair_name}_probability.npy"
            )

            # =================================================
            # STEP 1: Prediction
            # =================================================

            prediction_arguments = [

                "--before",
                str(before),

                "--after",
                str(after),

                "--out",
                str(prediction_output),
            ]

            prediction_result = run_python_script(
                "predict_change.py",
                prediction_arguments
            )

            # Print stdout
            if prediction_result.stdout:
                print(
                    prediction_result.stdout
                )

            # Check prediction failure
            if prediction_result.returncode != 0:

                print()
                print("PREDICTION FAILED")

                if prediction_result.stderr:
                    print(
                        prediction_result.stderr
                    )

                results.append({
                    "Region": region,
                    "Before": before_date,
                    "After": after_date,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Dice": np.nan,
                    "IoU": np.nan,
                    "Status": "Prediction Failed",
                })

                continue

            # -------------------------------------------------
            # Check prediction file
            # -------------------------------------------------

            if not prediction_file.exists():

                print()
                print(
                    "PREDICTION FILE NOT FOUND:"
                )

                print(
                    prediction_file
                )

                results.append({
                    "Region": region,
                    "Before": before_date,
                    "After": after_date,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Dice": np.nan,
                    "IoU": np.nan,
                    "Status": "Prediction File Missing",
                })

                continue

            # =================================================
            # STEP 2: Evaluation
            # =================================================

            evaluation_arguments = [

                "--prediction",
                str(prediction_file),

                "--ground-truth",
                str(ground_truth),

                "--out",
                str(prediction_output),
            ]

            evaluation_result = run_python_script(
                "evaluate_change.py",
                evaluation_arguments
            )

            # Print evaluation output
            if evaluation_result.stdout:
                print(
                    evaluation_result.stdout
                )

            # Check evaluation failure
            if evaluation_result.returncode != 0:

                print()
                print("EVALUATION FAILED")

                if evaluation_result.stderr:
                    print(
                        evaluation_result.stderr
                    )

                results.append({
                    "Region": region,
                    "Before": before_date,
                    "After": after_date,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Dice": np.nan,
                    "IoU": np.nan,
                    "Status": "Evaluation Failed",
                })

                continue

            # =================================================
            # STEP 3: Extract metrics
            # =================================================

            metrics = parse_metrics(
                evaluation_result.stdout
            )

            results.append({

                "Region": region,

                "Before": before_date,

                "After": after_date,

                "Accuracy": metrics["Accuracy"],

                "Precision": metrics["Precision"],

                "Recall": metrics["Recall"],

                "Dice": metrics["Dice"],

                "IoU": metrics["IoU"],

                "Status": "Success",
            })

            print()
            print("PAIR EVALUATION SUCCESSFUL")

    # =========================================================
    # Save all pair results
    # =========================================================

    df = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR /
        "all_test_pair_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # =========================================================
    # Successful evaluations
    # =========================================================

    successful = df[
        df["Status"] == "Success"
    ].copy()

    # =========================================================
    # Region-wise summary
    # =========================================================

    if len(successful) > 0:

        region_summary = (

            successful
            .groupby("Region")
            [
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "Dice",
                    "IoU",
                ]
            ]
            .mean()
            .reset_index()
        )

        region_summary_path = (
            OUTPUT_DIR /
            "region_summary.csv"
        )

        region_summary.to_csv(
            region_summary_path,
            index=False
        )

        print()
        print("=" * 70)
        print("REGION-WISE AVERAGE PERFORMANCE")
        print("=" * 70)

        print(
            region_summary.to_string(
                index=False
            )
        )

        # =================================================
        # Overall average
        # =================================================

        overall = successful[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "Dice",
                "IoU",
            ]
        ].mean()

        print()
        print("=" * 70)
        print("OVERALL TEST PERFORMANCE")
        print("=" * 70)

        print(
            f"Accuracy  : "
            f"{overall['Accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{overall['Precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{overall['Recall']:.4f}"
        )

        print(
            f"Dice / F1 : "
            f"{overall['Dice']:.4f}"
        )

        print(
            f"IoU       : "
            f"{overall['IoU']:.4f}"
        )

    else:

        print()
        print("=" * 70)
        print("NO SUCCESSFUL EVALUATIONS")
        print("=" * 70)

    # =========================================================
    # Final summary
    # =========================================================

    print()
    print("=" * 70)
    print("FULL TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Total temporal pairs attempted : "
        f"{total_pairs}"
    )

    print(
        f"Successful evaluations         : "
        f"{len(successful)}"
    )

    print()
    print("Results saved:")

    print(
        csv_path
    )

    if len(successful) > 0:

        print(
            OUTPUT_DIR /
            "region_summary.csv"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()