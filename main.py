
"""
Handwritten Digit Recognition using Random Forest
and Decision Tree Classification.

Main execution file for the complete ML pipeline.
"""

import os

from src.data_loader import load_data, split_data
from src.model_training import create_models, train_models
from src.evaluation import evaluate_models

from src.visualization import (
    plot_model_comparison,
    plot_confusion_matrices
)

from src.utils import (
    save_models,
    save_results,
    save_classification_reports
)


# ==========================================
# PROJECT DIRECTORY CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
GRAPH_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():
    """Execute the complete machine learning pipeline."""

    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION PROJECT")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1] Loading dataset...")

    X, y = load_data()

    print("Total Samples:", len(X))
    print("Number of Features:", X.shape[1])
    print("Number of Classes:", len(set(y)))

    # Step 2: Split dataset
    print("\n[2] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # Step 3: Create and train models
    print("\n[3] Training machine learning models...")

    models = create_models()

    trained_models = train_models(
        models,
        X_train,
        y_train
    )

    # Step 4: Evaluate models
    print("\n[4] Evaluating models...")

    results_df, reports, confusion_matrices = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    print("\nMODEL PERFORMANCE COMPARISON")
    print("-" * 60)
    print(results_df.round(4))

    # Step 5: Save trained models
    print("\n[5] Saving trained models...")

    save_models(
        trained_models,
        MODEL_DIR
    )

    # Step 6: Save evaluation results
    print("\n[6] Saving evaluation results...")

    save_results(
        results_df,
        os.path.join(
            REPORT_DIR,
            "model_comparison_results.csv"
        )
    )

    save_classification_reports(
        reports,
        REPORT_DIR
    )

    # Step 7: Generate visualizations
    print("\n[7] Generating visualizations...")

    plot_model_comparison(
        results_df,
        GRAPH_DIR
    )

    plot_confusion_matrices(
        confusion_matrices,
        GRAPH_DIR
    )

    # Step 8: Identify best model
    best_model = results_df["Accuracy"].idxmax()

    print("\nBEST PERFORMING MODEL:", best_model)

    print(
        "Best Accuracy:",
        round(
            results_df.loc[best_model, "Accuracy"] * 100,
            2
        ),
        "%"
    )

    # Final message
    print("\n" + "=" * 60)
    print("PROJECT EXECUTED SUCCESSFULLY!")
    print("=" * 60)


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
