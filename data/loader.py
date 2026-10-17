"""Dataset loader for AWID3 (CSV edition) or the synthetic sample.

Input:   config dict (dataset choice, path, column mapping, label values, subsampling, split).
Process: reads the CSV header, prints the columns actually present, resolves each canonical
         field to the first matching candidate source column (warning on misses), reads only the
         mapped columns (optionally the first N rows of each file), coerces types, keeps only the
         Normal + attack classes and binarises the label (0 = normal, 1 = deauth).
         `split()` then makes a deterministic train/test split seeded from the config.
Output:  `load()` -> DataFrame with canonical columns
         [time, inter_arrival, type_subtype, seq, reason_code, rssi, src, dst, label];
         `split()` -> (train_df, test_df).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["time", "inter_arrival", "seq", "reason_code", "rssi"]


def _files(path):
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files or not files[0].exists():
        raise FileNotFoundError(f"No CSV found at {p}. See README 'Getting AWID3'.")
    return files


def _number(s):
    """First numeric token of each cell; AWID3 RSSI cells can hold several antennas ('-61,-63')."""
    return pd.to_numeric(s.str.extract(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", expand=False), errors="coerce")


def _subtype(v):
    """tshark writes type/subtype as '12' or '0x000c' depending on version."""
    v = str(v).strip()
    try:
        return int(v, 16) if v.lower().startswith("0x") else float(v)
    except ValueError:
        return np.nan


def resolve_columns(header, cfg):
    """Map present source columns -> canonical names; warn on canonical fields with no match."""
    mapping = {}
    for canon, candidates in cfg["columns"].items():
        col = next((c for c in candidates if c in header), None)
        if col is None:
            print(f"[loader] WARNING: none of {candidates} found for '{canon}' — feature will be imputed")
        else:
            mapping[col] = canon
    if "label" not in mapping.values():
        raise ValueError("No label column found; add its name under columns.label in config.yaml")
    return mapping


def load(cfg):
    files = _files(cfg["paths"][cfg["dataset"]])
    header = list(pd.read_csv(files[0], nrows=0).columns)
    print(f"[loader] dataset={cfg['dataset']}  files={len(files)}  columns found ({len(header)}):")
    print("  " + ", ".join(header))
    mapping = resolve_columns(header, cfg)
    print("[loader] using: " + ", ".join(f"{v}<-{k}" for k, v in mapping.items()))

    n = cfg["loader"].get("max_rows_per_file")
    df = pd.concat([pd.read_csv(f, usecols=list(mapping), dtype=str, nrows=n) for f in files],
                   ignore_index=True).rename(columns=mapping)

    for c in NUMERIC:
        if c in df:
            df[c] = _number(df[c])
    if "type_subtype" in df:
        df["type_subtype"] = df["type_subtype"].map(_subtype)

    lab = df["label"].str.strip()
    normal, attack = cfg["labels"]["normal"], cfg["labels"]["attack"]
    keep = lab.isin(normal + attack)
    df = df[keep].copy()
    df["label"] = lab[keep].isin(attack).astype(int)
    print(f"[loader] kept {len(df)} of {len(keep)} rows (normal={int((df.label == 0).sum())}, "
          f"attack={int(df.label.sum())})")
    return df.reset_index(drop=True)


def split(df, cfg):
    """Deterministic split. Shuffled splits are stratified; temporal splits keep time order."""
    shuffle = cfg["loader"]["shuffle_split"]
    return train_test_split(df, test_size=cfg["loader"]["test_size"], random_state=cfg["seed"],
                            shuffle=shuffle, stratify=df["label"] if shuffle else None)
