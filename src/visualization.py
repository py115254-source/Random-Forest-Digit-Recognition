
"""
Visualization Module
Creates model comparison graphs and confusion matrices.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def create_output_directory(path):
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_model_comparison(results_df, output_dir):
    """Create performance comparison charts."""

    create_output_directory(output_dir)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for metric in metrics:
        plt.figure(figsize=(9, 5))

        results_df[metric].plot(
            kind="bar",
            title=f"Model Comparison - {metric}",
            ylabel=metric,
            xlabel="Machine Learning Models"
        )

        plt.ylim(0, 1.05)
        plt.xticks(rotation=0)
        plt.tight_layout()

        file_path = os.path.join(
            output_dir, f"{metric.lower().replace(' ', '_')}_comparison.png"
        )

        plt.savefig(file_path, dpi=300)
        plt.close()


def plot_confusion_matrices(confusion_matrices, output_dir):
    """Generate confusion matrix heatmaps."""

    create_output_directory(output_dir)

    for name, matrix in confusion_matrices.items():
        plt.figure(figsize=(9, 7))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_")
        file_path = os.path.join(
            output_dir, f"{filename}_confusion_matrix.png"
        )

        plt.savefig(file_path, dpi=300)
        plt.close()
