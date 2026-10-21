"""Detection-triggered response layer.

Input:   one detection at a time: attacker MAC (transmitter address, if known), confidence score,
         detector name, frame timestamp; config `mitigation` + outputs_dir.
Process: drops detections below min_score; appends a structured JSON alert to
         outputs/alerts.jsonl; for each newly seen MAC calls a pluggable blacklist hook. The
         default hook records the MAC in outputs/blocklist.txt and prints the action it WOULD
         take. MACs in `protected_macs` (your BSSID) are never blocklisted.
Output:  alerts.jsonl, blocklist.txt, console messages; respond() returns True if alerted.

REAL-WORLD LIMITATION: software cannot force a victim NIC to ignore a well-formed deauth frame.
Deauth/disassoc frames are unauthenticated management frames in legacy 802.11, and the attacker
usually SPOOFS the AP's own MAC — so "blocking the attacker" by MAC is weak at best and blocking
the spoofed BSSID would cut off the AP itself. This module is the detection-triggered response
layer (alerting, logging, operator/WIPS hooks); the true protocol fix is 802.11w / Protected
Management Frames (PMF), mandatory in WPA3, which cryptographically protects these frames.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def _is_unicast(mac):
    """Group (broadcast/multicast) addresses have the low bit of the first octet set; a frame
    claiming one as its transmitter is forged and there is nothing meaningful to block."""
    try:
        return not int(mac.split(":")[0], 16) & 1
    except ValueError:
        return False


class Responder:
    def __init__(self, cfg, blacklist_hook=None):
        out = Path(cfg["paths"]["outputs_dir"])
        out.mkdir(parents=True, exist_ok=True)
        m = cfg["mitigation"]
        self.alert_log = out / "alerts.jsonl"
        self.blocklist = out / "blocklist.txt"
        self.min_score = m["min_score"]
        self.protected = {mac.lower() for mac in m.get("protected_macs", [])}
        self.hook = blacklist_hook or self.record_blocklist
        self.blocked = set(self.blocklist.read_text().split()) if self.blocklist.exists() else set()
        self.alerts = 0

    def respond(self, mac, score, detector, frame_time=None):
        if score < self.min_score:
            return False
        mac = mac.lower() if isinstance(mac, str) and mac != "unknown" else None
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_time": frame_time,
            "attacker_mac": mac,
            "score": round(float(score), 4),
            "detector": detector,
        }
        with self.alert_log.open("a") as fh:
            fh.write(json.dumps(alert) + "\n")
        self.alerts += 1
        if mac and _is_unicast(mac) and mac not in self.blocked:
            self.blocked.add(mac)
            self.hook(mac)
        return True

    def record_blocklist(self, mac):
        """Default blacklist hook — a stub. Swap in a real one via Responder(cfg, blacklist_hook=f)."""
        if mac in self.protected:
            print(f"[responder] NOT blocking {mac}: protected MAC (likely the spoofed BSSID). "
                  "Recommend enabling 802.11w/PMF on the AP.")
            return
        with self.blocklist.open("a") as fh:
            fh.write(mac + "\n")
        print(f"[responder] WOULD block {mac}: e.g. add to AP deny-list "
              f"(hostapd deny_mac_file) and notify the WIPS operator")
