"""Command-line entry point.

Input:   subcommand (train | evaluate | run-live) and --config path (default config.yaml).
Process: train    — load -> extract shared features -> split -> fit + save the ML detectors.
         evaluate — same load/extract/split (same seed => same test rows) -> rule-based detector
                    from current config thresholds + saved ML detectors -> metrics, table, plots
                    -> responder fed with the configured detector's test-set detections.
         run-live — hands off to the live capture stub.
Output:  models/*.joblib, outputs/{comparison.csv, *.png, alerts.jsonl, blocklist.txt}.
"""
import argparse
import sys
from pathlib import Path

import yaml

from data import loader
from detectors import ml
from detectors.rule_based import RuleBasedDetector
from evaluate import metrics
from features.extract import FEATURES, extract
from mitigation.responder import Responder


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def prepare(cfg):
    """Features are extracted on the full time-ordered stream BEFORE splitting, so windowed
    features keep their temporal context; both splits then share one feature definition."""
    feats = extract(loader.load(cfg), cfg)
    return loader.split(feats, cfg)


def cmd_train(cfg):
    train, _ = prepare(cfg)
    return ml.train_all(train[FEATURES], train["label"], cfg)


def cmd_evaluate(cfg):
    _, test = prepare(cfg)
    X, y = test[FEATURES], test["label"]
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    rule = RuleBasedDetector(cfg["rules"])
    rule.save(models_dir / "rule_based.joblib")  # persisted only so model size is comparable
    detectors = [rule] + ml.load_all(cfg)
    results = [metrics.evaluate_detector(d, X, y, models_dir / f"{d.name}.joblib") for d in detectors]
    table = metrics.compare(results, cfg["paths"]["outputs_dir"])

    det = next(d for d in detectors if d.name == cfg["mitigation"]["detector"])
    responder = Responder(cfg)
    for (_, row), s in zip(test.iterrows(), det.score(X)):
        responder.respond(row["src"], s, det.name, float(row["time"]))
    print(f"[responder] {responder.alerts} alerts from {det.name} -> {responder.alert_log}")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wi-Fi deauthentication attack detection")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train and save the ML detectors")
    sub.add_parser("evaluate", help="compare all detectors on the test split")
    live = sub.add_parser("run-live", help="live capture (STUB)")
    live.add_argument("--iface", default="wlan0mon")
    live.add_argument("--detector", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "train":
        cmd_train(cfg)
    elif a.cmd == "evaluate":
        cmd_evaluate(cfg)
    else:
        from live.sniff_stub import run
        return run(cfg, a.iface, a.detector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
