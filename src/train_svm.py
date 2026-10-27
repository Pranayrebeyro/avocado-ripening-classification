"""Train an SVM classifier using color and texture features."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from feature_extraction import extract_image_features


def load_features(dataframe):
    """Extract features and labels from a dataframe of image paths."""
    features = []
    labels = []

    for row in dataframe.itertuples(index=False):
        features.append(extract_image_features(row.path))
        labels.append(row.class_name)

    return np.asarray(features), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing path and class_name columns.",
    )
    parser.add_argument(
        "--output",
        default="models/classical_svm.joblib",
        help="Output path for the trained SVM pipeline.",
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(args.csv)
    required_columns = {"path", "class_name"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"CSV must contain these columns: {sorted(required_columns)}"
        )

    if dataframe.empty:
        raise ValueError("The input CSV contains no images.")

    X, y = load_features(dataframe)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X, y)

    predictions = model.predict(X)
    print("Training-set accuracy:", round(accuracy_score(y, predictions), 4))
    print("Training-set macro F1:", round(f1_score(y, predictions, average="macro"), 4))
    print(classification_report(y, predictions, zero_division=0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
