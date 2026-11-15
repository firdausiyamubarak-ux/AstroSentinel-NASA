"""AstroSentinel dashboard.  Run:  python run.py   then   streamlit run app.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support

HERE = Path(__file__).parent
st.set_page_config(page_title="AstroSentinel", layout="wide")
st.title("AstroSentinel")
st.caption("Tamper-aware AI health monitoring for astronauts. All data is SIMULATED "
           "(based on NASA Human Research Program hazards).")

if not (HERE / "results.csv").exists():
    st.error("Run `python run.py` first to create results.csv and test_stream.csv.")
    st.stop()

res = pd.read_csv(HERE / "results.csv")
stream = pd.read_csv(HERE / "test_stream.csv")
W = 15
UNITS = {"hr": "bpm", "spo2": "%", "resp": "breaths/min", "temp": "°C",
         "sbp": "mmHg", "co2": "ppm", "rad": "µSv/h"}

# headline metrics
pr, rc, _, _ = precision_recall_fscore_support(res.truth, res.pred, labels=[1, 2], zero_division=0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Medical recall", f"{rc[0]:.0%}")
c2.metric("Medical precision", f"{pr[0]:.0%}")
c3.metric("Attack recall", f"{rc[1]:.0%}")
c4.metric("Attack precision", f"{pr[1]:.0%}")

day = st.slider("Day of mission (test period)", 8, 14, 8)
chans = st.multiselect("Vitals to show", list(UNITS), default=["hr", "spo2", "co2"])
d0 = (day - 8) * 1440
seg = stream.iloc[d0:d0 + 1440]
rseg = res[(res.t_min >= d0) & (res.t_min < d0 + 1440)]

fig, axes = plt.subplots(len(chans) + 1, 1, figsize=(11, 2.1 * (len(chans) + 1)), sharex=True,
                         gridspec_kw={"height_ratios": [1] * len(chans) + [0.45]})
axes = np.atleast_1d(axes)
hours = (np.arange(len(seg))) / 60
for ax, c in zip(axes[:-1], chans):
    ax.plot(hours, seg[c].values, lw=1, color="#1f5fbf")
    ax.set_ylabel(f"{c} ({UNITS[c]})", fontsize=8)
    for _, row in rseg.iterrows():
        a = (row.t_min - d0) / 60
        if row.truth == 1:
            ax.axvspan(a, a + W / 60, color="#f0a500", alpha=0.22, lw=0)
        elif row.truth == 2:
            ax.axvspan(a, a + W / 60, color="#d62728", alpha=0.22, lw=0)
ax = axes[-1]
for _, row in rseg.iterrows():
    a = (row.t_min - d0) / 60
    col = {0: "#e6e6e6", 1: "#f0a500", 2: "#d62728"}[row.pred]
    ax.axvspan(a, a + W / 60, color=col, lw=0)
ax.set_yticks([])
ax.set_ylabel("AstroSentinel", fontsize=8)
ax.set_xlabel("hour of day")
ax.set_xlim(0, 24)
st.pyplot(fig)
st.caption("Shading on vitals = ground truth (amber medical event, red cyber attack). "
           "Bottom strip = what AstroSentinel decided (grey normal, amber medical alert, red attack).")

st.subheader("Alerts on this day")
alerts = rseg[rseg.pred > 0][["t_min", "label"]].copy()
alerts["time"] = ((alerts.t_min - d0) // 60).astype(int).astype(str).str.zfill(2) + ":" + \
    ((alerts.t_min - d0) % 60).astype(int).astype(str).str.zfill(2)
st.dataframe(alerts[["time", "label"]], hide_index=True, width="stretch")
