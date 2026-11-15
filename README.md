# AstroSentinel

Tamper-aware AI health monitoring for astronauts. Built for the NASA Space Apps Challenge 2026
challenge **"Create Health Monitoring Software for Astronauts on Space Missions"**.

AstroSentinel watches astronaut vitals, raises **medical alerts** for real health events, and
separately flags **cyber attacks** on the sensor data (spoofing, drift, replay, stuck sensors,
injected spikes, and masking of a real emergency). Telling the two apart is the core idea: a crew
should never be misled by forged data, and a real emergency should never be hidden by it.

## IMPORTANT disclosure: simulated data
NASA's challenge page lists background articles but no raw astronaut dataset. All vitals here are
**simulated** (`simulate.py`). The modelled hazards come from NASA's Human Research Program:
space radiation, cabin CO2 / hypoxia, infection (fever) and cardiac events. Numbers are plausible
but are **not real crew data**, so the reported accuracy applies to this simulation only.

## How it works
1. **Simulator** (`simulate.py`): 14 days, 1 sample/min, 7 vitals (HR, SpO2, respiration, temperature,
   systolic BP, cabin CO2, radiation dose rate) with daily rhythm, exercise, sleep and 13 medical events.
2. **Attack injector** (`attacks.py`): 7 tampering attacks in the test days.
3. **Detector** (`detect.py`), trained on days 0-6, tested on days 7-13:
   - *Physiology layer*: personal time-of-day baseline + Isolation Forest on baseline deviations.
   - *Integrity layer*: each vital is predicted from the others (Ridge regression); data that
     breaks physiological consistency is flagged. Plus flatline and exact-replay checks.
   - Decision: integrity problem -> **ATTACK**; otherwise physiology problem -> **MEDICAL ALERT**.
4. **Dashboard** (`app.py`, Streamlit).

## Run it
```
pip install -r requirements.txt
python run.py          # trains, evaluates, writes results.csv and test_stream.csv
streamlit run app.py   # dashboard
```

## Results (15-minute windows, held-out test days)
Averaged over 5 random simulations: medical recall ~96%, precision ~88%; attack recall ~93%,
precision ~97%. Limits: simulated data, hand-designed attacks, one detector design. Not a medical
device.

## References (official NASA / space agency sources)
- NASA Human Research Program, hazards of human spaceflight: https://www.nasa.gov/hrp/hazards/
- NASA Human Research Program, risks of human spaceflight: https://www.nasa.gov/hrp/risks/
- NASA Human Research Program, ongoing research: https://www.nasa.gov/hrp/ongoing-hrp-research/
- Canadian Space Agency, monitoring radiation for astronaut health:
  https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp
- Python libraries: NumPy, pandas, scikit-learn, matplotlib, Streamlit (open source).
- AI acknowledgement: The code and text in this project were written with the help of an AI assistant (Claude, by Anthropic). The machine-learning methods are Isolation Forest and Ridge regression (scikit-learn). No AI-generated images or videos are used.

License: Apache 2.0.
