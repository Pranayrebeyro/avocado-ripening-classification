"""Evaluate classification predictions using standard metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred, output_dir, model_name):
    """Calculate metrics, save a report, and plot a confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "correct_predictions": int(np.sum(np.asarray(y_true) == np.asarray(y_pred))),
        "total_images": int(len(y_true)),
        "classification_report": report,
    }

    report_path = output_dir / f"{model_name}_evaluation.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    figure_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"\nModel: {model_name}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")
    print(f"Saved metrics: {report_path}")
    print(f"Saved confusion matrix: {figure_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing true_label and predicted_label columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--model-name",
        default="model",
        help="Name used for output filenames.",
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(args.csv)
    required = {"true_label", "predicted_label"}
    if not required.issubset(dataframe.columns):
        raise ValueError(
            "CSV must contain columns: true_label, predicted_label"
        )

    if dataframe.empty:
        raise ValueError("The evaluation CSV contains no predictions.")

    evaluate_predictions(
        dataframe["true_label"].astype(str).tolist(),
        dataframe["predicted_label"].astype(str).tolist(),
        args.output_dir,
        args.model_name,
    )


if __name__ == "__main__":
    main()
