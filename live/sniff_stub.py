"""
##########################################################################################
#  STUB — NOT YET COMPLETE. Live capture is future work; this file only sketches the loop. #
##########################################################################################

Live-mode pipeline sketch.

Input:   a monitor-mode wireless interface (e.g. wlan0mon), a trained detector from models/,
         config dict.
Process: scapy sniff loop -> packet_to_row (canonical frame fields) -> rolling buffer ->
         features.extract (the SAME feature code as offline) -> detector.score on the newest
         frame -> mitigation.responder on detection.
Output:  alerts/blocklist via the Responder; console messages.

Importing this module needs no hardware and no scapy: scapy is imported inside run().

TODO before this is usable:
  - run with root + a NIC in monitor mode; add channel hopping or pin to the AP's channel
  - incremental features instead of re-extracting the whole buffer per packet (O(buffer))
  - drop frames from other BSSIDs; handle radiotap fields missing on some drivers
  - test against a real deauth flood in an isolated lab network you own
"""
from collections import deque
from pathlib import Path

import pandas as pd

from detectors.base import Detector
from features.extract import FEATURES, extract
from mitigation.responder import Responder

BUFFER_FRAMES = 500


def packet_to_row(pkt, Dot11, RadioTap, Dot11Deauth, Dot11Disas):
    """Map a scapy 802.11 packet to the canonical fields used by data.loader."""
    d = pkt[Dot11]
    reason = None
    if pkt.haslayer(Dot11Deauth):
        reason = pkt[Dot11Deauth].reason
    elif pkt.haslayer(Dot11Disas):
        reason = pkt[Dot11Disas].reason
    return {
        "time": float(pkt.time),
        "type_subtype": (d.type << 4) | d.subtype,  # same encoding as tshark wlan.fc.type_subtype
        "seq": (d.SC >> 4) if d.SC is not None else None,
        "reason_code": reason,
        "rssi": getattr(pkt[RadioTap], "dBm_AntSignal", None) if pkt.haslayer(RadioTap) else None,
        "src": d.addr2,
        "dst": d.addr1,
    }


def run(cfg, iface, detector_name=None):
    try:
        from scapy.all import Dot11, Dot11Deauth, Dot11Disas, RadioTap, sniff
    except ImportError:
        print("[live] STUB: scapy is not installed (pip install scapy). Live mode not available.")
        return 1
    name = detector_name or cfg["mitigation"]["detector"]
    path = Path(cfg["paths"]["models_dir"]) / f"{name}.joblib"
    if not path.exists():
        print(f"[live] STUB: {path} not found — run `python main.py train` first.")
        return 1
    detector, responder = Detector.load(path), Responder(cfg)
    buffer = deque(maxlen=BUFFER_FRAMES)

    def handle(pkt):
        if not pkt.haslayer(Dot11):
            return
        buffer.append(packet_to_row(pkt, Dot11, RadioTap, Dot11Deauth, Dot11Disas))
        newest = extract(pd.DataFrame(buffer), cfg).iloc[[-1]]
        score = float(detector.score(newest[FEATURES])[0])
        responder.respond(newest["src"].iat[0], score, detector.name, newest["time"].iat[0])

    print(f"[live] STUB: sniffing on {iface} with {name} (Ctrl+C to stop)")
    try:
        sniff(iface=iface, prn=handle, store=False)
    except (PermissionError, OSError) as e:
        print(f"[live] STUB: capture failed ({e}). Needs root and a monitor-mode interface.")
        return 1
    return 0
