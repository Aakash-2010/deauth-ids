"""Common detector interface shared by the rule-based and ML detectors.

Input:   the shared feature matrix df[features.extract.FEATURES] (and labels, for fit).
Process: defines fit (optional; no-op by default), predict and score, plus joblib persistence,
         so every detector is called, timed and sized in exactly the same way.
Output:  predict -> np.ndarray of 0/1 labels; score -> np.ndarray of confidences in [0, 1].
"""
from abc import ABC, abstractmethod

import joblib


class Detector(ABC):
    name = "detector"

    def fit(self, X, y):
        return self

    @abstractmethod
    def predict(self, X):
        """Return 0/1 labels, one per row of X."""

    def score(self, X):
        """Confidence that each row is an attack. Default: the hard label."""
        return self.predict(X).astype(float)

    def save(self, path):
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path):
        return joblib.load(path)
