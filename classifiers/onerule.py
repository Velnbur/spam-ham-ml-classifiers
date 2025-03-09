"""
Provides OneRule classifier.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class OneRuleClassifier:
    """
    One rule classifier, marks each sample of data as spam
    if there are any words from vocabluary.
    """

    def __init__(self, max_words: int = 10):
        self.max_words = max_words
        self.vectorizer = CountVectorizer(stop_words="english")
        self.vocabulary: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Finds the most used words from "spam" values and adds them to vocabulary.
        """

        spam = X[y == "spam"]
        vectorized = self.vectorizer.fit_transform(spam)

        # sum all vectorized emails by each word
        sum_vectorized = np.sum(vectorized.toarray(), axis=0, keepdims=False)
        # and find N the most used words.
        most_used = np.argpartition(sum_vectorized, -self.max_words)[-self.max_words :]

        self.vocabulary = self.vectorizer.get_feature_names_out()[most_used]

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict for each value in X whenether it's spam by most used
        words in vocabulary.
        """

        mask = np.zeros(len(X))
        for x in X:
            for word in self.vocabulary:
                mask = np.logical_or(np.strings.find(x, str(word)) != -1, mask)

        y = np.where(mask, 1.0, 0.0)
        return y
