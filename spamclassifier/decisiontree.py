"""
Implementation of Spam/Ham classifier using decision trees.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import TypeAlias
import multiprocessing as mp
import time

from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class _Node:
    """
    Node of the decision tree
    """

    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None

    info_gain: float | None = None


@dataclass
class Leaf:
    """
    Class for leaf of the decision tree.
    """

    label: float

    @staticmethod
    def from_y(y: np.ndarray) -> Leaf:
        """
        Returns the Leaf with label with most samples of particular class.
        """
        unique, counts = np.unique(y, return_counts=True)
        return Leaf(label=unique[np.argmax(counts)])

    def __repr__(self):
        return f"Leaf(label={self.label})"


Node: TypeAlias = _Node | Leaf


@dataclass
class _SplitData:
    feature: int
    threshold: float
    X_left: np.ndarray
    X_right: np.ndarray
    y_left: np.ndarray
    y_right: np.ndarray
    info_gain: float

    @staticmethod
    def default() -> _SplitData:
        return _SplitData(
            feature=-1,
            threshold=-1,
            X_left=np.array([]),
            X_right=np.array([]),
            y_left=np.array([]),
            y_right=np.array([]),
            info_gain=0,
        )


@dataclass
class FindBestSplit:
    X: np.ndarray
    y: np.ndarray
    thresholds_max_num: int | None = None

    def __call__(self, feature_idx: int) -> tuple[float, _SplitData]:
        best_split = _SplitData.default()
        max_info_gain = sys.float_info.min

        feature_values = self.X[:, feature_idx]

        if self.thresholds_max_num is not None:
            thresholds = np.linspace(
                feature_values.min(), feature_values.max(), self.thresholds_max_num
            )
        else:
            thresholds = np.unique(feature_values)

        for threshold in thresholds:
            X_left, X_right, y_left, y_right = _split_data(
                self.X, self.y, feature_idx, threshold
            )

            # skip if one of datasets is empty
            if len(X_left) == 0 or len(X_right) == 0:
                continue

            info_gain = _information_gain(self.y, y_left, y_right)

            if info_gain > max_info_gain:
                best_split = _SplitData(
                    feature=feature_idx,
                    threshold=threshold,
                    X_left=X_left,
                    X_right=X_right,
                    y_left=y_left,
                    y_right=y_right,
                    info_gain=info_gain,
                )
                max_info_gain = info_gain

        return max_info_gain, best_split


class DecisionTreeClassifier:
    """
    Decision tree for spam/ham classifier

    Args:
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the tree.
        vectorizer (CountVectorizer): The vectorizer used to transform the input data.
        thresholds_max_num (int): The maximum number of thresholds to consider for each feature.
    """

    root: Node | None
    min_samples_split: int
    max_depth: int
    thresholds_max_num: int

    def __init__(
        self,
        min_samples_split=2,
        max_depth=10,
        vectorizer=None,
        max_features=100,
    ):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.vectorizer = vectorizer

        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                stop_words="english", max_features=max_features
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree classifier to the training data.
        """

        self.vectorizer.fit(X)
        X = self.vectorizer.transform(X).toarray()
        y = np.where(y == "spam", 1.0, 0.0)

        self.root = self._build_tree(X, y)

    def predict(self, X: np.ndarray):
        """
        Predict the class labels for the input data.
        """

        X = self.vectorizer.transform(X).toarray()

        return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.ndarray):
        """
        Predict the class label for a single input sample.
        """

        node = self.root
        while not isinstance(node, Leaf):
            if node is None:
                raise ValueError(
                    "Node is None, something went wrong during fitting the data"
                )

            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.label

    def _build_tree(self, X: np.ndarray, y: np.ndarray, curr_depth: int = 0):
        num_samples, num_features = X.shape

        if num_samples < self.min_samples_split or curr_depth > self.max_depth:
            return Leaf.from_y(y)

        best_split = self._best_split(X, y, num_samples, num_features)

        if best_split.info_gain <= 0:
            return Leaf.from_y(y)

        left_subtree = self._build_tree(
            best_split.X_left, best_split.y_left, curr_depth + 1
        )
        right_subtree = self._build_tree(
            best_split.X_right, best_split.y_right, curr_depth + 1
        )

        return _Node(
            feature=best_split.feature,
            threshold=best_split.threshold,
            left=left_subtree,
            right=right_subtree,
            info_gain=best_split.info_gain,
        )

    def _best_split(
        self,
        X: np.array,
        y: np.array,
        num_samples: int,
        num_features: int,
    ) -> _SplitData:
        """
        Find the best split (threshold, feature) for the given data.
        """

        max_info_gain = sys.float_info.min
        best_split = _SplitData.default()

        with mp.Pool() as pool:
            split_fn = FindBestSplit(X, y)

            for info_gain, best_split in tqdm(
                pool.imap_unordered(split_fn, range(num_features)),
                total=num_features,
                disable=True,
            ):
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split = best_split

        return best_split


def _split_data(
    X: np.array, y: np.array, feature_idx: int, threshold: float
) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Split the data into two subsets based on the given feature and threshold.
    """

    mask = X[:, feature_idx] <= threshold
    X_left = X[mask]
    X_right = X[~mask]
    y_left = y[mask]
    y_right = y[~mask]

    return X_left, X_right, y_left, y_right


def _information_gain(y: np.array, y_left: np.array, y_right: np.array) -> float:
    """
    Calculate the information gain for the given split.
    """

    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)

    parent_entropy = _entropy(y)
    left_entropy = _entropy(y_left)
    right_entropy = _entropy(y_right)

    return (
        parent_entropy - (weight_left * left_entropy) - (weight_right * right_entropy)
    )


def _entropy(y: np.array) -> float:
    """
    Calculate the entropy of the given dataset.
    """

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))
