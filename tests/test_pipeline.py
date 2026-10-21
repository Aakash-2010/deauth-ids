"""Smoke test: generate sample -> extract -> train -> evaluate -> holdout runs end to end.

Input:   none (writes everything under pytest's tmp_path).
Process: generates two synthetic CSVs in a folder (the second is the holdout file), points a copy
         of config.yaml at them, runs `train`, `evaluate` and `holdout`, and checks the feature
         contract, the split, that the holdout file is never read by train/evaluate, and outputs.
Output:  pass/fail.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from data import loader  # noqa: E402
from data.generator import generate  # noqa: E402
from features.extract import FEATURES  # noqa: E402


def test_pipeline_end_to_end(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    generate(seed=7).to_csv(frames / "part_1.csv", index=False)
    generate(seed=8).to_csv(frames / "part_2.csv", index=False)

    cfg = main.load_config(ROOT / "config.yaml")
    cfg["dataset"] = "sample"
    cfg["loader"]["holdout_files"] = 1
    cfg["paths"].update(sample=str(frames), models_dir=str(tmp_path / "models"),
                        outputs_dir=str(tmp_path / "outputs"))

    assert [f.name for f in loader.select_files(cfg, "main")] == ["part_1.csv"]
    assert [f.name for f in loader.select_files(cfg, "holdout")] == ["part_2.csv"]

    train, val, test = main.prepare(cfg)
    assert list(train.columns[:len(FEATURES)]) == FEATURES
    assert not train[FEATURES].isna().any().any()
    assert set(train["label"]) == {0, 1}
    assert len(train) + len(val) + len(test) == len(loader.load(cfg, "main"))

    assert len(main.cmd_train(cfg)) == len(cfg["models"])
    table = main.cmd_evaluate(cfg)
    holdout = main.cmd_holdout(cfg)

    assert list(table.index) == ["rule_based", *cfg["models"]]
    assert table["f1"].between(0, 1).all()
    assert (table[["tn", "fp", "fn", "tp"]].sum(axis=1) == len(test)).all()
    assert holdout["tp"].sum() > 0
    out = tmp_path / "outputs"
    for stage in ["validation", "test", "holdout"]:
        for f in ["comparison.csv", "scores_bar.png", "latency_vs_f1.png"]:
            assert (out / f"{stage}_{f}").exists(), f"{stage}_{f}"
    assert (out / "alerts.jsonl").exists()
