"""Shared feature extraction: canonical per-frame rows -> the ONE feature matrix every detector uses.

Input:   canonical frame DataFrame from data.loader.load() (or the live sniffer), plus config
         (features.window_seconds, features.rssi_window_frames).
Process: sorts frames by time, imputes missing raw fields, and derives the enriched features
         below. All windowed features look only BACKWARD in time, so they are computable online.
Output:  DataFrame with columns FEATURES + ['label', 'src', 'time']. Detectors receive exactly
         df[FEATURES] — the rule-based and ML detectors see identical rows and columns, so any
         performance difference is attributable to the detection method alone.
"""
import numpy as np
import pandas as pd

FEATURES = ["is_mgmt_deauth", "is_mgmt_disassoc", "deauth_rate_window", "seq_delta",
            "inter_arrival_time", "rssi", "rssi_deviation", "reason_code"]
DEAUTH, DISASSOC = 12, 10   # 802.11 management subtypes (type 0): 0x0c, 0x0a
SEQ_MOD = 4096              # 12-bit sequence number space


def is_subtype(type_subtype, code):
    """Input: type_subtype codes. Process: equality with the given subtype (NaN -> no).
    Output: 0/1 int Series (is_mgmt_deauth for 12, is_mgmt_disassoc for 10)."""
    return (type_subtype == code).astype(int)


def deauth_rate_window(time, kill, window_seconds):
    """Input: frame times (s, sorted) and 0/1 deauth-or-disassoc flags.
    Process: time-based rolling sum over the trailing `window_seconds` (includes current frame).
    Output: count of deauth/disassoc frames seen in the window ending at each frame."""
    s = pd.Series(kill.to_numpy(), index=pd.to_datetime(time.to_numpy(), unit="s"))
    return s.rolling(pd.Timedelta(seconds=window_seconds)).sum().to_numpy()


def seq_delta(seq, src):
    """Input: 12-bit sequence numbers and transmitter MACs.
    Process: per source, distance from the expected next number (prev + 1), modulo 4096, taking
    the shorter way round so retransmits/backward jumps are also small-vs-large correctly.
    Output: 0 for an in-order frame; large values mean a foreign counter (spoofed MAC).
    First frame of a source, or missing seq -> 0."""
    prev = seq.groupby(src).shift()
    d = (seq - prev - 1) % SEQ_MOD
    return np.minimum(d, SEQ_MOD - d).fillna(0)


def inter_arrival_time(inter_arrival, time):
    """Input: raw inter-arrival column (may be missing) and frame times.
    Process: use the raw value, else the time difference to the previous frame; clip at 0.
    Output: seconds since the previous captured frame."""
    return inter_arrival.fillna(time.diff()).fillna(0).clip(lower=0)


def rssi(raw_rssi, src):
    """Input: raw RSSI (dBm, may be missing) and transmitter MACs.
    Process: impute missing values with the source's median, then the global median, then -100.
    Output: RSSI in dBm with no NaNs."""
    filled = raw_rssi.fillna(raw_rssi.groupby(src).transform("median"))
    return filled.fillna(filled.median()).fillna(-100.0)


def rssi_deviation(rssi_values, src, n):
    """Input: imputed RSSI and transmitter MACs.
    Process: |RSSI - mean of that source's previous n frames|. A spoofer transmitting from a
    different physical position than the real owner of the MAC shows up as a deviation.
    Output: dB deviation; 0 for a source's first frame."""
    # ponytail: per-group Python lambda; vectorise with groupby().rolling() if AWID3 runs are slow
    base = rssi_values.groupby(src).transform(lambda s: s.shift().rolling(n, min_periods=1).mean())
    return (rssi_values - base).abs().fillna(0)


def extract(raw, cfg):
    df = raw.copy()
    for c in ["time", "inter_arrival", "type_subtype", "seq", "reason_code", "rssi"]:
        if c not in df:
            df[c] = np.nan
    if df["time"].isna().all():
        df["time"] = df["inter_arrival"].fillna(0).cumsum()
    df["time"] = df["time"].ffill().fillna(0)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    src = df["src"].fillna("unknown") if "src" in df else pd.Series("unknown", index=df.index)
    f = cfg["features"]

    out = pd.DataFrame(index=df.index)
    out["is_mgmt_deauth"] = is_subtype(df["type_subtype"], DEAUTH)
    out["is_mgmt_disassoc"] = is_subtype(df["type_subtype"], DISASSOC)
    out["deauth_rate_window"] = deauth_rate_window(
        df["time"], out["is_mgmt_deauth"] + out["is_mgmt_disassoc"], f["window_seconds"])
    out["seq_delta"] = seq_delta(df["seq"], src)
    out["inter_arrival_time"] = inter_arrival_time(df["inter_arrival"], df["time"])
    out["rssi"] = rssi(df["rssi"], src)
    out["rssi_deviation"] = rssi_deviation(out["rssi"], src, f["rssi_window_frames"])
    out["reason_code"] = df["reason_code"].fillna(0)   # 0 = not a deauth/disassoc frame
    out["label"] = df["label"].astype(int) if "label" in df else 0
    out["src"] = src
    out["time"] = df["time"]
    return out
