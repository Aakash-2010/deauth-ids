"""ML deauth detectors: RandomForest, GradientBoosting and an MLP ("lightweight neural network").

Input:   shared feature matrix df[FEATURES] + labels; model list/kwargs and seed from config.
Process: builds each sklearn estimator with random_state = config seed, fits it on the SAME
         feature matrix the rule-based detector reads, and saves/loads it with joblib.
         The MLP is wrapped with a StandardScaler (gradient-based nets need scaled inputs); the
         scaler is part of the model, so the input features stay identical across detectors.
Output:  fitted MLDetector objects (predict -> 0/1, score -> P(attack)) and models/<name>.joblib.
"""
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detectors.base import Detector

ESTIMATORS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier,
}


class MLDetector(Detector):
    def __init__(self, name, model):
        self.name, self.model = name, model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).astype(int)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


def build(name, params, seed):
    if name not in ESTIMATORS:
        raise KeyError(f"Unknown model '{name}'; choose from {list(ESTIMATORS)}")
    est = ESTIMATORS[name](random_state=seed, **(params or {}))
    if name == "mlp":
        est = make_pipeline(StandardScaler(), est)
    return MLDetector(name, est)


def train_all(X, y, cfg):
    """Fit every model listed in config and save each to models_dir/<name>.joblib."""
    out = Path(cfg["paths"]["models_dir"])
    out.mkdir(parents=True, exist_ok=True)
    detectors = []
    for name, params in cfg["models"].items():
        det = build(name, params, cfg["seed"]).fit(X, y)
        det.save(out / f"{name}.joblib")
        print(f"[ml] trained + saved {name}")
        detectors.append(det)
    return detectors


def load_all(cfg):
    out = Path(cfg["paths"]["models_dir"])
    return [Detector.load(out / f"{name}.joblib") for name in cfg["models"]]
