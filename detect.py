"""AstroSentinel detector: separates real MEDICAL events from cyber ATTACKS on sensor data.

Layers
1. Physiology model  : Isolation Forest on 15-min window features, trained on normal windows.
2. Integrity layer   : (a) cross-sensor consistency - each vital is predicted from the others
                       with a Ridge model; large residuals mean the data breaks physiology.
                       (b) flatline check, (c) replay check (exact repeat of past samples).
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

W = 15
CH = ["hr", "spo2", "resp", "temp", "sbp", "co2", "rad"]
CORE = ["hr", "spo2", "resp", "sbp", "co2"]  # channels used for integrity checks
NORMAL, MEDICAL, ATTACK = 0, 1, 2


def _windows(a):
    nw = len(a) // W
    return a[:nw * W].reshape(nw, W, -1)


DAY = 1440


def _feats(Z):
    """Window features of baseline-deviation z-scores."""
    w = _windows(Z)
    return np.concatenate([w.mean(1), w.std(1)], axis=1)


def _profile(df):
    """Personal time-of-day baseline: median of each vital per minute-of-day over normal data."""
    X = df[CH].values.astype(float).copy()
    X[df["event"].values.astype(bool)] = np.nan
    n = (len(X) // DAY) * DAY
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = np.nanmedian(X[:n].reshape(-1, DAY, len(CH)), axis=0)
    k = 31
    pad = np.concatenate([P[-k:], P, P[:k]])
    ker = np.ones(k) / k
    P = np.stack([np.convolve(pad[:, j], ker, mode="same") for j in range(P.shape[1])], 1)
    return P[k:-k]


class AstroSentinel:
    def fit(self, df):
        X = df[CH].values
        ev = _windows(df[["event"]].values)[:, :, 0].mean(1) > 0
        self.profile = _profile(df)
        base = np.tile(self.profile, (len(X) // DAY + 1, 1))[:len(X)]
        resid = X - base
        resid[df["event"].values.astype(bool)] = np.nan
        self.z_sd = np.nanstd(resid, axis=0)
        F = _feats(self._z(X))
        self.scaler = StandardScaler().fit(F[~ev])
        self.iforest = IsolationForest(n_estimators=300, random_state=0).fit(
            self.scaler.transform(F[~ev]))
        self.ph_thr = np.percentile(
            self.iforest.decision_function(self.scaler.transform(F[~ev])), 1)
        # per-vital deviation limits from normal windows (mean |z| per window)
        zm = np.abs(_windows(self._z(X)).mean(1))
        self.ch_thr = 1.25 * zm[~ev].max(axis=0)

        # integrity: predict each core vital from the others
        idx = [CH.index(c) for c in CORE]
        C = X[:, idx]
        self.models, self.res_sd = {}, {}
        for j, c in enumerate(CORE):
            others = [k for k in range(len(CORE)) if k != j]
            m = Ridge(alpha=1.0).fit(C[:, others], C[:, j])
            self.models[c] = (m, others)
            self.res_sd[c] = (C[:, j] - m.predict(C[:, others])).std()
        self.int_thr = np.percentile(self._integrity(X), 99.7)

        # flatline threshold from normal window std
        sd = _windows(C).std(1)
        self.flat_thr = 0.2 * np.percentile(sd, 5, axis=0)

        # replay memory
        self.seen = set(map(tuple, np.round(C, 3)))
        return self

    def _z(self, X):
        base = np.tile(self.profile, (len(X) // DAY + 1, 1))[:len(X)]
        return (X - base) / self.z_sd

    def _integrity(self, X):
        idx = [CH.index(c) for c in CORE]
        C = X[:, idx]
        z = []
        for j, c in enumerate(CORE):
            m, others = self.models[c]
            z.append(np.abs(C[:, j] - m.predict(C[:, others])) / self.res_sd[c])
        z = np.stack(z, 1)  # (n, ch)
        return _windows(z).mean(1).max(1)

    def predict(self, df):
        X = df[CH].values
        F = _feats(self._z(X))
        ph = self.iforest.decision_function(self.scaler.transform(F))
        zm = np.abs(_windows(self._z(X)).mean(1))
        dev = (zm > self.ch_thr).any(1)
        integ = self._integrity(X)

        idx = [CH.index(c) for c in CORE]
        C = X[:, idx]
        flat = (_windows(C).std(1) < self.flat_thr).any(1)

        keys = list(map(tuple, np.round(C, 3)))
        rep = np.zeros(len(keys) // W)
        seen = set(self.seen)
        for w in range(len(rep)):
            ks = keys[w * W:(w + 1) * W]
            rep[w] = np.mean([k in seen for k in ks])
            seen.update(ks)
        replay = rep > 0.5

        n = len(ph)
        pred = np.zeros(n, dtype=int)
        sub = np.array(["normal"] * n, dtype=object)
        for i in range(n):
            if replay[i]:
                pred[i], sub[i] = ATTACK, "replay"
            elif flat[i]:
                pred[i], sub[i] = ATTACK, "stuck sensor"
            elif integ[i] > self.int_thr:
                pred[i], sub[i] = ATTACK, "inconsistent data (spoof/drift/injection)"
            elif ph[i] < self.ph_thr or dev[i]:
                pred[i], sub[i] = MEDICAL, "medical alert"
        return pred, sub, ph, integ

    def physiology_only(self, df):
        """Baseline: AI anomaly detector with no integrity layer (flags anything unusual)."""
        Z = self._z(df[CH].values)
        F = _feats(Z)
        dev = (np.abs(_windows(Z).mean(1)) > self.ch_thr).any(1)
        return ((self.iforest.decision_function(self.scaler.transform(F)) < self.ph_thr) | dev).astype(int)


def window_truth(df, attack_mask):
    ev = _windows(df[["event"]].values)[:, :, 0].mean(1) > 0.5
    at = _windows(attack_mask.reshape(-1, 1))[:, :, 0].mean(1) > 0.5
    y = np.zeros(len(ev), dtype=int)
    y[ev] = MEDICAL
    y[at] = ATTACK
    return y
