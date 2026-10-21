"""Command-line entry point.

Input:   subcommand (train | evaluate | holdout | run-live) and --config path (default config.yaml).
Process: Data is partitioned twice. (1) The last `loader.holdout_files` files are a holdout that
         train/evaluate never read. (2) The remaining files are split train / validation / test.
         train    — load main files -> extract shared features -> split -> fit + save the ML
                    detectors -> report all detectors on VALIDATION (tune thresholds here).
         evaluate — same load/extract/split (same seed => same rows) -> rule-based detector from
                    current config + saved ML detectors -> metrics on TEST -> responder.
         holdout  — load ONLY the holdout files -> extract -> metrics on HOLDOUT -> responder.
                    The models have never seen these files, not even for tuning.
         run-live — hands off to the live capture stub.
Output:  models/*.joblib, outputs/{validation,test,holdout}_{comparison.csv, *.png},
         outputs/{alerts.jsonl, blocklist.txt}.
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
    features keep their temporal context; all splits then share one feature definition."""
    feats = extract(loader.load(cfg, part="main"), cfg)
    return loader.split(feats, cfg)


def detectors(cfg):
    """Rule-based detector (from current config thresholds) + the saved ML detectors."""
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    rule = RuleBasedDetector(cfg["rules"])
    rule.save(models_dir / "rule_based.joblib")  # persisted only so model size is comparable
    return [rule] + ml.load_all(cfg)


def report(cfg, df, stage, respond=False):
    X, y = df[FEATURES], df["label"]
    dets = detectors(cfg)
    md = Path(cfg["paths"]["models_dir"])
    results = [metrics.evaluate_detector(d, X, y, md / f"{d.name}.joblib") for d in dets]
    table = metrics.compare(results, cfg["paths"]["outputs_dir"], stage)
    if respond:
        det = next(d for d in dets if d.name == cfg["mitigation"]["detector"])
        responder = Responder(cfg)
        scores = det.score(X)
        hits = df.assign(score=scores)[scores >= responder.min_score]
        for mac, s, t in zip(hits["src"], hits["score"], hits["time"]):
            responder.respond(mac, s, det.name, float(t))
        print(f"[responder] {responder.alerts} alerts from {det.name} -> {responder.alert_log}")
    return table


def cmd_train(cfg):
    train, val, _ = prepare(cfg)
    trained = ml.train_all(train[FEATURES], train["label"], cfg)
    report(cfg, val, "validation")
    return trained


def cmd_evaluate(cfg):
    _, _, test = prepare(cfg)
    return report(cfg, test, "test", respond=True)


def cmd_holdout(cfg):
    feats = extract(loader.load(cfg, part="holdout"), cfg)
    return report(cfg, feats, "holdout", respond=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wi-Fi deauthentication attack detection")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train the ML detectors; report on the validation split")
    sub.add_parser("evaluate", help="compare all detectors on the test split")
    sub.add_parser("holdout", help="compare all detectors on the never-seen holdout files")
    live = sub.add_parser("run-live", help="live capture (STUB)")
    live.add_argument("--iface", default="wlan0mon")
    live.add_argument("--detector", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "run-live":
        from live.sniff_stub import run
        return run(cfg, a.iface, a.detector)
    {"train": cmd_train, "evaluate": cmd_evaluate, "holdout": cmd_holdout}[a.cmd](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
