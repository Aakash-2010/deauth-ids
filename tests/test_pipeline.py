"""Smoke test: generate sample -> extract -> train -> evaluate runs end to end.

Input:   none (writes everything under pytest's tmp_path).
Process: generates a fresh synthetic CSV, points a copy of config.yaml at it, runs the `train`
         and `evaluate` commands, checks the feature contract and the outputs.
Output:  pass/fail.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from data.generator import generate  # noqa: E402
from features.extract import FEATURES  # noqa: E402


def test_pipeline_end_to_end(tmp_path):
    csv = tmp_path / "frames.csv"
    generate(seed=7).to_csv(csv, index=False)

    cfg = main.load_config(ROOT / "config.yaml")
    cfg["dataset"] = "sample"
    cfg["paths"].update(sample=str(csv), models_dir=str(tmp_path / "models"),
                        outputs_dir=str(tmp_path / "outputs"))

    train, test = main.prepare(cfg)
    assert list(train.columns[:len(FEATURES)]) == FEATURES
    assert not train[FEATURES].isna().any().any()
    assert set(train["label"]) == {0, 1}

    assert len(main.cmd_train(cfg)) == len(cfg["models"])
    table = main.cmd_evaluate(cfg)

    assert list(table.index) == ["rule_based", *cfg["models"]]
    assert table["f1"].between(0, 1).all()
    assert (table[["tn", "fp", "fn", "tp"]].sum(axis=1) == len(test)).all()
    for f in ["comparison.csv", "scores_bar.png", "latency_vs_f1.png", "alerts.jsonl"]:
        assert (tmp_path / "outputs" / f).exists(), f
