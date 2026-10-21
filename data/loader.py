"""Dataset loader for AWID3 (CSV edition) or the synthetic sample.

Input:   config dict (dataset choice, path, column mapping, label values, subsampling, split).
Process: reads the CSV header, prints the columns actually present, resolves each canonical
         field to the first matching candidate source column (warning on misses), reads only the
         mapped columns (optionally the first N rows of each file), coerces types, keeps only the
         Normal + attack classes and binarises the label (0 = normal, 1 = deauth).
         The last `holdout_files` files are reserved: load(part="main") never reads them,
         load(part="holdout") reads only them. `split()` then makes a deterministic
         train / validation / test split of the main part, seeded from the config.
Output:  `load()` -> DataFrame with canonical columns
         [time, inter_arrival, type_subtype, seq, reason_code, rssi, src, dst, label];
         `split()` -> (train_df, val_df, test_df).
"""
from pathlib import Path

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["time", "inter_arrival", "seq", "reason_code", "rssi", "fc_type", "fc_subtype"]


def _files(path):
    p = Path(path)
    # numeric-aware order: Deauth_2.csv before Deauth_10.csv
    files = sorted(p.glob("*.csv"), key=lambda f: [int(t) if t.isdigit() else t
                                                  for t in re.split(r"(\d+)", f.name)]) if p.is_dir() else [p]
    if not files or not files[0].exists():
        hint = ("run `python data/generator.py` to create it" if "sample" in str(p)
                else "see README 'Getting AWID3'")
        raise FileNotFoundError(f"No CSV found at {p}; {hint}.")
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
    mapping, missing = {}, []
    for canon, candidates in cfg["columns"].items():
        col = next((c for c in candidates if c in header), None)
        if col is None:
            missing.append((canon, candidates))
        else:
            mapping[col] = canon
    found = set(mapping.values())
    derivable = {"type_subtype"} if {"fc_type", "fc_subtype"} <= found else set()
    derivable |= {"fc_type", "fc_subtype"} if "type_subtype" in found else set()
    for canon, candidates in missing:
        if canon not in derivable:
            print(f"[loader] WARNING: none of {candidates} found for '{canon}' — feature will be imputed")
    if "label" not in mapping.values():
        raise ValueError("No label column found; add its name under columns.label in config.yaml")
    return mapping


def select_files(cfg, part):
    """part='main' -> every file except the last `holdout_files`; part='holdout' -> only those."""
    files = _files(cfg["paths"][cfg["dataset"]])
    k = cfg["loader"].get("holdout_files") or 0
    if k and len(files) <= k:
        print(f"[loader] WARNING: only {len(files)} file(s); holdout disabled for this dataset")
        k = 0
    if part == "holdout":
        if not k:
            raise ValueError("No holdout files: set loader.holdout_files and use a directory with more files")
        return files[-k:]
    return files[:-k] if k else files


def load(cfg, part="main"):
    files = select_files(cfg, part)
    header = list(pd.read_csv(files[0], nrows=0).columns)
    print(f"[loader] dataset={cfg['dataset']}  part={part}  files={len(files)} "
          f"({files[0].name} .. {files[-1].name})  columns found ({len(header)}):")
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
    elif {"fc_type", "fc_subtype"} <= set(df):
        # Subtype alone is ambiguous (type 1 / subtype 12 is CTS, not deauth), so combine.
        df["type_subtype"] = df["fc_type"] * 16 + df["fc_subtype"]
        print("[loader] derived type_subtype from fc_type*16 + fc_subtype")
    df = df.drop(columns=["fc_type", "fc_subtype"], errors="ignore")

    lab = df["label"].str.strip()
    normal, attack = cfg["labels"]["normal"], cfg["labels"]["attack"]
    keep = lab.isin(normal + attack)
    df = df[keep].copy()
    df["label"] = lab[keep].isin(attack).astype(int)
    print(f"[loader] kept {len(df)} of {len(keep)} rows (normal={int((df.label == 0).sum())}, "
          f"attack={int(df.label.sum())})")
    return df.reset_index(drop=True)


def split(df, cfg):
    """Deterministic train/val/test split. Shuffled = stratified; temporal = time order kept."""
    if df["label"].nunique() < 2:
        raise ValueError(f"Only one class present ({df['label'].iloc[0]} = "
                         f"{'deauth' if df['label'].iloc[0] else 'normal'}); cannot train/test a "
                         "binary detector. Add CSVs that contain attack frames (see README).")
    lc, seed = cfg["loader"], cfg["seed"]
    shuffle, val, test = lc["shuffle_split"], lc["val_size"], lc["test_size"]

    def cut(d, size):
        return train_test_split(d, test_size=size, random_state=seed, shuffle=shuffle,
                                stratify=d["label"] if shuffle else None)

    rest, test_df = cut(df, test)
    train_df, val_df = cut(rest, val / (1 - test))   # val as a fraction of the whole
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[split] {name:5s} {len(d):>9} rows, {int(d['label'].sum()):>7} attacks")
        if d["label"].nunique() < 2:
            msg = f"{name} split has a single class (e.g. attacks only occur late in the capture)"
            if name == "train":
                raise ValueError(msg + "; set loader.shuffle_split: true")
            print(f"[split] WARNING: {msg}")
    return train_df, val_df, test_df
