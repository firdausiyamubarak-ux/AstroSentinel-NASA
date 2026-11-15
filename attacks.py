"""Cyber-attack injector: tampers with the telemetry stream of the monitoring software."""
import numpy as np
from simulate import DAY, EVENTS, DURATION


def inject(df, seed=1):
    """Return (tampered copy, attack mask array, attack log). Attacks only in test days 7-13."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = np.zeros(len(df), dtype=int)
    log = []

    def mark(s, e, kind):
        mask[s:e] = 1
        log.append((kind, s, e))

    # 1. spoofed heart rate -> false alarm
    s = 7 * DAY + 300; e = s + 90
    out.loc[s:e - 1, "hr"] = 140 + rng.normal(0, 1.0, e - s)
    mark(s, e, "spoof_hr")

    # 2. slow drift on SpO2 (stealthy)
    s = 8 * DAY + 900; e = s + 180
    out.loc[s:e - 1, "spo2"] -= np.linspace(0, 7, e - s)
    mark(s, e, "drift_spo2")

    # 3. stuck / flatlined sensors
    s = 9 * DAY + 200; e = s + 120
    for c in ("hr", "resp"):
        out.loc[s:e - 1, c] = out.loc[s, c]
    mark(s, e, "flatline")

    # 4. replay of old data from 2 days earlier
    s = 10 * DAY + 300; e = s + 90
    cols = ["hr", "spo2", "resp", "temp", "sbp", "co2", "rad"]
    out.loc[s:e - 1, cols] = df.loc[s - 2 * DAY:e - 2 * DAY - 1, cols].values
    mark(s, e, "replay")

    # 5. masking a real hypoxia event (day 11): attacker hides vitals with normal-looking values
    hs = 11 * DAY + 400; he = hs + DURATION["hypoxia"]
    for c in ("hr", "spo2", "resp", "sbp"):
        base = df.loc[hs - DAY:he - DAY - 1, c].values
        out.loc[hs:he - 1, c] = base + rng.normal(0, 0.15, he - hs)
    mark(hs, he, "mask_event")

    # 6. injected spikes
    s = 12 * DAY + 200; e = s + 60
    idx = np.arange(s, e)
    pick = idx[rng.random(len(idx)) < 0.25]
    out.loc[pick, "sbp"] += 60
    mark(s, e, "spike_sbp")

    # 7. second spoof, shorter
    s = 13 * DAY + 900; e = s + 60
    out.loc[s:e - 1, "hr"] = 135 + rng.normal(0, 1.0, e - s)
    mark(s, e, "spoof_hr")

    return out, mask, log
