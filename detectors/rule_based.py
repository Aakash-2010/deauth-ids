"""Rule-based (signature/threshold) deauth detector. No training.

Input:   shared feature matrix df[FEATURES]; thresholds from config `rules`.
Process: a frame is flagged only if it IS a deauth/disassoc frame (packet-type gate) AND its
         anomaly conditions fire — combined with OR (`mode: any`) or AND (`mode: all`):
           1. deauth_rate_window > deauth_rate_threshold   (flood in the time window)
           2. seq_delta          > seq_delta_threshold     (sequence counter foreign to the MAC)
           3. rssi_deviation     > rssi_deviation_threshold (sender not where the MAC's owner is)
Output:  predict -> 0/1 labels; score -> fraction of the three conditions met (0 for non-deauth).
"""
import numpy as np

from detectors.base import Detector


class RuleBasedDetector(Detector):
    name = "rule_based"

    def __init__(self, rules):
        self.rules = dict(rules)

    def _conditions(self, X):
        r = self.rules
        kill = ((X["is_mgmt_deauth"] + X["is_mgmt_disassoc"]) > 0).to_numpy()
        conds = np.column_stack([
            X["deauth_rate_window"].to_numpy() > r["deauth_rate_threshold"],
            X["seq_delta"].to_numpy() > r["seq_delta_threshold"],
            X["rssi_deviation"].to_numpy() > r["rssi_deviation_threshold"],
        ])
        return kill, conds

    def predict(self, X):
        kill, conds = self._conditions(X)
        hit = conds.all(axis=1) if self.rules["mode"] == "all" else conds.any(axis=1)
        return (kill & hit).astype(int)

    def score(self, X):
        kill, conds = self._conditions(X)
        return conds.mean(axis=1) * kill
