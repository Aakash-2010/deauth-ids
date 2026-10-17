# Wi-Fi Deauthentication Attack Detection Using Machine Learning

A head-to-head comparison of a **rule-based** detector and **ML** detectors (Random Forest,
Gradient Boosting, MLP) for 802.11 deauthentication attacks. Every detector reads the **same
feature matrix**, built by the same code, so any difference in performance comes from the
detection method alone. A mitigation module responds when an attack is detected.

Stack: Python 3, pandas, numpy, scikit-learn, joblib, matplotlib, PyYAML (and scapy for the live stub only).

## Setup

```bash
cd deauth-ids
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start (synthetic sample, no download needed)

`config.yaml` ships with `dataset: sample`, which points at `data/sample/sample_frames.csv`
(about 2000 frames, roughly 14% deauth).

```bash
python main.py train       # fit + save the ML detectors to models/
python main.py evaluate    # compare all detectors; writes outputs/
python -m pytest -q        # smoke test
```

To regenerate the sample: `python data/generator.py [--seed 42]`.

> The synthetic sample only shows that the pipeline works. **Don't report its numbers** in the
> paper; use AWID3 for that.

## Getting AWID3

1. Request access to AWID3 through the AWID project page at the University of the Aegean
   (icsdweb.aegean.gr/awid). Download the **CSV** edition.
2. Copy the CSVs for the **Deauth** attack into `data/awid3/`. This can be one file or a
   folder of files; files are read in sorted filename order. The folder is git-ignored.
3. In `config.yaml`, set `dataset: awid3`.
4. Run `python main.py train`. The loader prints every column it finds and warns about any
   canonical field it can't match. If a column name differs in your copy, add it to the
   candidate list under `columns:`. The first candidate found in the file is used.
5. For a quick development run on the large files, set `loader.max_rows_per_file` (for example
   `200000`). This keeps the first N rows of each file instead of sampling rows at random,
   because random sampling would break the time windows the features depend on.

Only the `Normal` and `Deauth` rows are kept, and the label becomes binary (0 = normal,
1 = deauth). To also count disassociation floods as attacks, add `Disas` to `labels.attack`.

## CLI

| Command | What it does |
|---|---|
| `python main.py train` | load → extract features → split → train and save RF, GB and MLP |
| `python main.py evaluate` | same split (same seed) → rule-based plus saved ML detectors → metrics table, plots, responder |
| `python main.py run-live --iface wlan0mon` | live-capture **stub**. Needs scapy, root and a monitor-mode NIC |

All commands accept `--config path.yaml`.

Outputs, in `outputs/`:

- `comparison.csv`: accuracy, precision, recall, F1, TN/FP/FN/TP, latency (ms per sample) and model size (KB)
- `scores_bar.png` and `latency_vs_f1.png`
- `alerts.jsonl` and `blocklist.txt`, written by the responder

The data is imbalanced, so **F1 and the confusion matrix are the primary metrics**.

## Shared features (`features/extract.py`)

| Feature | Meaning |
|---|---|
| `is_mgmt_deauth`, `is_mgmt_disassoc` | frame subtype is 12 (deauth) or 10 (disassoc) |
| `deauth_rate_window` | deauth and disassoc frames in the trailing `features.window_seconds` |
| `seq_delta` | gap from the expected next sequence number for the same transmitter (mod 4096) |
| `inter_arrival_time` | seconds since the previous frame |
| `rssi`, `rssi_deviation` | signal strength, and how far it is from that transmitter's rolling mean |
| `reason_code` | deauth/disassoc reason code (0 for other frames) |

Features are computed on the full time-ordered stream **before** the train/test split, and they
only look backward in time. This means the same code also works online in the live stub.

## Module → system-diagram block

| Diagram block | Module |
|---|---|
| Data acquisition | `data/loader.py` (AWID3 / sample), `data/generator.py` (synthetic), `live/sniff_stub.py` (live) |
| Feature extraction (shared) | `features/extract.py` |
| Detection: common interface | `detectors/base.py` |
| Detection: rule-based | `detectors/rule_based.py` |
| Detection: ML | `detectors/ml.py` |
| Mitigation / response | `mitigation/responder.py` |
| Evaluation | `evaluate/metrics.py` |
| Orchestration / CLI | `main.py`, configured by `config.yaml` |

## Notes for the paper

- **Fairness.** Both kinds of detector receive exactly `df[FEATURES]`. The MLP's
  StandardScaler is part of the MLP model, not part of the shared features.
- **Rule thresholds** are read from `config.yaml` each time `evaluate` runs, so you don't need
  to retrain after changing them.
- **Split choice.** With `shuffle_split: true` (stratified random split), frames from the same
  attack burst can land in both train and test, which makes the ML scores look better than they
  are. For headline results, consider `shuffle_split: false`, a temporal split that trains on
  earlier frames and tests on later ones. Check that the test portion still contains attacks.
- **Reproducibility.** The single `seed` controls the generator, the split and every model.
- **Mitigation limit.** Software can't make a victim's NIC ignore a well-formed deauth frame.
  Attackers usually spoof the AP's own MAC, so blocking by MAC is weak. That's why the AP's
  BSSID goes in `mitigation.protected_macs`. The real protocol fix is **802.11w / PMF**, which
  is mandatory in WPA3.
