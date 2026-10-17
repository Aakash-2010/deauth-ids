"""Synthetic 802.11 frame generator (stand-in for AWID3 so the pipeline runs with no download).

Input:   seed and duration (CLI: `python data/generator.py [--seed N] [--out PATH]`).
Process: simulates one AP + six stations for ~60 s: beacons, QoS data + ACKs, a few legitimate
         deauth/disassoc frames (stations leaving, an AP "kick everyone" burst that looks like a
         flood), and three aireplay-ng-style deauth attack bursts at high/medium/low rate that
         SPOOF the AP's MAC and run their own sequence counter from a different RSSI position.
         Sequence numbers are assigned per transmitter in time order, like real hardware.
Output:  CSV using the same AWID3 column names as the real dataset (so the loader code path is
         identical), labelled Normal / Deauth — default data/sample/sample_frames.csv (~2000 rows).

The sample is for verifying the pipeline end to end, NOT for reporting results.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AP = "aa:bb:cc:00:00:01"
STATIONS = [f"11:22:33:44:55:{i:02x}" for i in range(1, 7)]
BCAST = "ff:ff:ff:ff:ff:ff"
BEACON, DATA, ACK, DEAUTH, DISASSOC = 8, 40, 29, 12, 10
# (start s, duration s, frames/s): high-rate, medium-rate, and a stealthy low-rate burst
ATTACKS = [(10.0, 2.0, 80), (25.0, 2.0, 40), (50.0, 5.0, 8)]


def generate(seed=42, duration=60.0):
    rng = np.random.default_rng(seed)
    base_rssi = {AP: -45, "attacker": -52, **{s: int(rng.integers(-70, -50)) for s in STATIONS}}
    ev = []  # (time, subtype, src, dst, rssi_key, seq_key, reason, label)

    def add(t, sub, src, dst, rkey, skey=None, reason=None, label="Normal"):
        ev.append((t, sub, src, dst, rkey, skey if skey else src, reason, label))

    for t in np.arange(0, duration, 0.2):                        # beacons
        add(t + rng.uniform(0, 0.005), BEACON, AP, BCAST, AP)
    for sta in STATIONS:                                          # data both ways + ACKs
        for src, dst in ((sta, AP), (AP, sta)):
            t = rng.exponential(1.0)
            while t < duration:
                add(t, DATA, src, dst, src)
                add(t + 0.0002, ACK, None, src, dst, skey="-")    # ACK: no TA, no seq
                t += rng.exponential(1.0)
    for sta in rng.choice(STATIONS, 3, replace=False):            # stations leaving
        t = rng.uniform(0, duration)
        add(t, DISASSOC, sta, AP, sta, reason=8)
        add(t + 0.001, DEAUTH, sta, AP, sta, reason=3)
    for i, sta in enumerate(STATIONS):                            # AP kicks everyone at t=40
        add(40 + i * 0.04, DEAUTH, AP, sta, AP, reason=4)
        add(40 + i * 0.04 + 0.01, DISASSOC, AP, sta, AP, reason=1)
    for start, dur, rate in ATTACKS:                              # spoofed deauth floods
        t = start
        while t < start + dur:
            victim = BCAST if rng.random() < 0.3 else rng.choice(STATIONS)
            reason = rng.choice([7, 1, 3], p=[0.8, 0.1, 0.1])
            add(t, DEAUTH, AP, victim, "attacker", skey="attacker", reason=reason, label="Deauth")
            t += rng.exponential(1.0 / rate)

    ev.sort(key=lambda e: e[0])
    counters, rows = {}, []
    for t, sub, src, dst, rkey, skey, reason, label in ev:
        seq = None
        if skey != "-":
            seq = counters.setdefault(skey, int(rng.integers(0, 4096)))
            counters[skey] = (seq + 1) % 4096
        rows.append({
            "frame.time_epoch": f"{1_600_000_000 + t:.6f}",
            "wlan.fc.type_subtype": sub,
            "wlan.seq": seq,
            "wlan.fixed.reason_code": reason,
            "radiotap.dbm_antsignal": int(round(base_rssi[rkey] + rng.normal(0, 2))),
            "wlan.ta": src,
            "wlan.ra": dst,
            "Label": label,
        })
    df = pd.DataFrame(rows)
    df.insert(1, "frame.time_delta", df["frame.time_epoch"].astype(float).diff().fillna(0).round(6))
    df["wlan.seq"] = df["wlan.seq"].astype("Int64")
    df["wlan.fixed.reason_code"] = df["wlan.fixed.reason_code"].astype("Int64")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "sample" / "sample_frames.csv"))
    a = ap.parse_args()
    df = generate(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}\n{df['Label'].value_counts().to_string()}")
