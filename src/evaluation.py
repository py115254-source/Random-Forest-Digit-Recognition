
"""
Model Evaluation Module
Evaluates trained models using classification metrics.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models and return results."""

    results = {}
    reports = {}
    confusion_matrices = {}

    for name, model in models.items():

        predictions = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, average="weighted"
            ),
            "Recall": recall_score(
                y_test, predictions, average="weighted"
            ),
            "F1 Score": f1_score(
                y_test, predictions, average="weighted"
            )
        }

        reports[name] = classification_report(
            y_test, predictions
        )

        confusion_matrices[name] = confusion_matrix(
            y_test, predictions
        )

    results_df = pd.DataFrame(results).T

    return results_df, reports, confusion_matrices
