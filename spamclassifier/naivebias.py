"""
Implementaion of Naive Bias for spam/ham classification
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBiasClassifier:
    """ """

    def __init__(self, max_words: int = 10):
        self.max_words = max_words
        self.vectorizer = CountVectorizer(stop_words="english")
        self.vocabulary: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Count probabilities for each word to appear in each class.
        """
        self.vectorizer.fit(X)

        spam = self.vectorizer.transform(X[y == "spam"]).toarray()
        ham = self.vectorizer.transform(X[y == "ham"]).toarray()

        sumed_spam = np.sum(spam, axis=0, keepdims=False)
        sumed_ham = np.sum(ham, axis=0, keepdims=False)

        total = sumed_spam.sum(keepdims=False)
        self.probs_spam = sumed_spam + 1 / (total + len(self.vectorizer.vocabulary_))
        self.class_spam_prob = len(spam) / len(X)

        total = sumed_ham.sum(keepdims=False)
        self.probs_ham = sumed_ham + 1 / (total + len(self.vectorizer.vocabulary_))
        self.class_ham_prob = len(ham) / len(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        vectorized = self.vectorizer.transform(X)

        probs = np.log(
            np.repeat(
                np.array([self.class_ham_prob, self.class_spam_prob]),
                repeats=len(X),
            ).reshape((-1, 2))
        )

        word_probs = np.array([self.probs_ham, self.probs_spam])

        probs += vectorized * np.log(word_probs).T

        return np.where(probs[:, 0] >= probs[:, 1], 0.0, 1.0)
