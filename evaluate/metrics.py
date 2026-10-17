"""Evaluation of every detector on the same held-out test rows.

Input:   fitted detectors, test feature matrix df[FEATURES], test labels, saved model paths.
Process: times predict() over the whole test set (averaged over repeats, then per sample),
         computes accuracy/precision/recall/F1 and the confusion matrix, reads model size on
         disk, builds a comparison table and two plots. Data is imbalanced, so F1 and the
         confusion matrix are the primary metrics; accuracy is reported for completeness only.
Output:  per-detector dict; outputs/comparison.csv; outputs/scores_bar.png;
         outputs/latency_vs_f1.png; the table printed to the console.
"""
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# validated categorical palette, fixed order: precision, recall, F1
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#2b2b2b", "#8a8a85"


def evaluate_detector(det, X, y, model_path, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        pred = det.predict(X)
    latency_ms = (time.perf_counter() - start) / repeats / len(X) * 1000
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "detector": det.name,
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "latency_ms": latency_ms,
        "size_kb": os.path.getsize(model_path) / 1024,
    }


def compare(results, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).set_index("detector")
    table.to_csv(out / "comparison.csv")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160,
                           "display.max_columns", None):
        print("\n=== Detector comparison (primary: F1 + confusion matrix) ===")
        print(table)
    _bar(table, out / "scores_bar.png")
    _scatter(table, out / "latency_vs_f1.png")
    return table


def _style(ax, title, pad=6):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=pad)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _bar(table, path):
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(table))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (m, c) in enumerate(zip(metrics, COLORS)):
        bars = ax.bar(x + (i - 1) * w, table[m], w - 0.02, label=m.upper() if m == "f1" else m.title(),
                      color=c)
        if m == "f1":  # selective direct labels: the primary metric only
            ax.bar_label(bars, fmt="%.3f", fontsize=8, color=INK, padding=2)
    ax.set_xticks(x, table.index)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score", color=INK)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(ax, "Precision, recall and F1 by detector (test set)", pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(table, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(table["latency_ms"], table["f1"], s=70, color=COLORS[0], edgecolors="white", linewidths=2)
    for i, (name, row) in enumerate(table.sort_values("latency_ms").iterrows()):
        ax.annotate(name, (row["latency_ms"], row["f1"]), textcoords="offset points",
                    xytext=(6, 8 if i % 2 == 0 else -14), fontsize=9, color=INK)  # alternate to avoid overlap
    ax.set_xscale("log")
    ax.set_xlabel("mean inference latency per sample (ms, log scale)", color=INK)
    ax.set_ylabel("F1", color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(max(0.0, table["f1"].min() - 0.1), 1.02)
    _style(ax, "Latency vs. F1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
