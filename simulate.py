"""Simulated astronaut vitals (1 sample / minute) with NASA-HRP-inspired medical events.

DISCLOSURE: all data here is SIMULATED. The hazards modelled (spaceflight radiation,
cabin CO2 / hypoxia, infection/fever, cardiac arrhythmia) come from NASA's Human
Research Program hazard and risk pages. Values are plausible but not real crew data.
"""
import numpy as np
import pandas as pd

CHANNELS = ["hr", "spo2", "resp", "temp", "sbp", "co2", "rad"]
DAY = 1440  # minutes per day

# (day, start minute inside day, kind). Days 0-6 = train, days 7-13 = test.
EVENTS = [
    (1, 700, "hypoxia"), (2, 300, "fever"), (3, 900, "arrhythmia"),
    (4, 500, "radiation"), (5, 800, "hypoxia"), (6, 200, "fever"),
    (7, 900, "arrhythmia"), (8, 300, "fever"), (9, 700, "hypoxia"),
    (10, 1000, "radiation"), (11, 400, "hypoxia"), (12, 800, "fever"),
    (13, 300, "radiation"),
]
DURATION = {"hypoxia": 180, "fever": 360, "arrhythmia": 40, "radiation": 120}


def _ar1(rng, n, phi, sigma):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _envelope(dur):
    k = np.arange(dur)
    return np.minimum(1.0, np.minimum(k + 1, dur - k) / (0.2 * dur))


def generate(days=14, seed=0):
    rng = np.random.default_rng(seed)
    n = days * DAY
    t = np.arange(n)
    hod = (t % DAY) / DAY
    circ = np.sin(2 * np.pi * (hod - 0.3))
    sleep = (hod < 0.27).astype(float)

    act = np.zeros(n)
    for d in range(days):
        s = d * DAY + 600 + int(rng.integers(-5, 5))  # crews follow a strict daily timeline
        act[s:s + 120] = 1.0
    act = np.convolve(act, np.ones(15) / 15, mode="same")

    co2_x = np.zeros(n)
    fever = np.zeros(n)
    arr = np.zeros(n)
    rad_m = np.ones(n)
    event = np.zeros(n, dtype=int)
    kind = np.array([""] * n, dtype=object)
    for day, off, k in EVENTS:
        if day >= days:
            continue
        s = day * DAY + off
        dur = DURATION[k]
        env = _envelope(dur)
        if k == "hypoxia":
            co2_x[s:s + dur] += 6000 * env
        elif k == "fever":
            fever[s:s + dur] += 1.8 * env
        elif k == "arrhythmia":
            arr[s:s + dur] += 40 * env
        elif k == "radiation":
            rad_m[s:s + dur] += 5 * env
        event[s:s + dur] = 1
        kind[s:s + dur] = k

    hyp = 0.0008 * co2_x  # SpO2 drop (%) caused by CO2 event
    co2 = 2800 + _ar1(rng, n, 0.98, 25) + co2_x
    spo2 = 98 - 0.8 * act - hyp + _ar1(rng, n, 0.7, 0.25)
    temp = 36.8 + 0.25 * circ + 0.4 * act + fever + _ar1(rng, n, 0.9, 0.03)
    hr = (68 + 6 * circ + 45 * act - 8 * sleep + 10 * fever + 3 * hyp + arr
          + _ar1(rng, n, 0.8, 1.2))
    resp = 14 + 0.12 * (hr - 68) + 0.9 * hyp + _ar1(rng, n, 0.7, 0.25)
    sbp = 118 + 0.35 * (hr - 68) + _ar1(rng, n, 0.8, 1.0)
    rad = 21 * rad_m + _ar1(rng, n, 0.8, 0.6)

    df = pd.DataFrame({"hr": hr, "spo2": spo2, "resp": resp, "temp": temp,
                       "sbp": sbp, "co2": co2, "rad": rad})
    df["event"] = event
    df["event_type"] = kind
    return df
