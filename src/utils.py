
"""
Utility Module
Handles model saving and results export.
"""

import os
import joblib


def save_models(models, output_dir):
    """Save trained machine learning models."""

    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"

        file_path = os.path.join(output_dir, filename)

        joblib.dump(model, file_path)

        print(f"Model saved: {file_path}")


def save_results(results_df, output_path):
    """Save model evaluation results as CSV."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results_df.to_csv(output_path, index=True)

    print(f"Results saved: {output_path}")


def save_classification_reports(reports, output_dir):
    """Save classification reports as text files."""

    os.makedirs(output_dir, exist_ok=True)

    for name, report in reports.items():
        filename = name.lower().replace(" ", "_") + "_report.txt"

        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(report)

        print(f"Report saved: {file_path}")
