import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from simulate import generate, DAY
from attacks import inject
from detect import AstroSentinel, window_truth, W, NORMAL, MEDICAL, ATTACK

df = generate(days=14, seed=0)
tampered, mask, log = inject(df)

train = df.iloc[:7 * DAY].reset_index(drop=True)
test = tampered.iloc[7 * DAY:].reset_index(drop=True)
test_mask = mask[7 * DAY:]

model = AstroSentinel().fit(train)
pred, sub, ph, integ = model.predict(test)
y = window_truth(test, test_mask)

names = ["normal", "medical", "attack"]
print(classification_report(y, pred, target_names=names, digits=3, zero_division=0))
print("confusion (rows=truth, cols=pred):\n", confusion_matrix(y, pred, labels=[0, 1, 2]))

print("\nPer-attack detection (any attack window flagged ATTACK):")
base = model.physiology_only(test)
for kind, s, e in log:
    s0, e0 = s - 7 * DAY, e - 7 * DAY
    w0, w1 = s0 // W, int(np.ceil(e0 / W))
    caught = (pred[w0:w1] == ATTACK).mean()
    flagged_base = base[w0:w1].mean()
    print(f"  {kind:12s} windows={w1 - w0:3d}  AstroSentinel={caught:4.0%}  physiology-only AI flagged={flagged_base:4.0%}")

res = pd.DataFrame({"t_min": np.arange(len(y)) * W, "truth": y, "pred": pred, "label": sub,
                    "physiology_score": ph, "integrity_score": integ})
res.to_csv("results.csv", index=False)
test.assign(attack=test_mask).to_csv("test_stream.csv", index=False)
print("\nsaved results.csv, test_stream.csv")
