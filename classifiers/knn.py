"""
This file provides the KNN classifier for spam/ham problem.
"""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class KNNClassifier:
    """
    K-Nearest Neighbors classifier for spam/ham problem.
    """

    X_train: None | np.ndarray
    y_train: None | np.ndarray
    k: int

    def __init__(self, k=3, vectorizer=None):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.vectorizer = vectorizer
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(stop_words="english")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store the training data.
        """

        self.X_train = self.vectorizer.fit_transform(X).toarray()
        self.y_train = np.where(y == "spam", 1.0, 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the test data.
        """
        X = self.vectorizer.transform(X).toarray()
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x: np.ndarray):
        """
        Predict the class label for a single data point.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not trained")

        distances = _euclidean_distance(self.X_train, x)
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]


def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between array of vectors, and one
    base vector.
    """
    return np.sqrt(np.sum(np.subtract(x1, x2) ** 2, axis=-1))
