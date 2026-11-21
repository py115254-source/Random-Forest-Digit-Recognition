
"""
Data Loader Module
Loads and prepares the handwritten digits dataset.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data():
    """Load the handwritten digits dataset."""
    digits = load_digits()

    X = digits.data
    y = digits.target

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
