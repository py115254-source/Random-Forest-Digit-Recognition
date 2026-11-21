
"""
Model Training Module
Trains Random Forest and Decision Tree classifiers.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def create_models():
    """Create machine learning models."""

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    return models


def train_models(models, X_train, y_train):
    """Train all models and return trained models."""

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models
