"""Synthetic checks for eegfeatures.py and leakage_test.py. Says nothing about real data."""
import numpy as np
import pandas as pd

from eegfeatures import FEATURE_NAMES, REGIONS, features_from_data
import leakage_test

rng = np.random.default_rng(0)
fs = 250
names = [c for chs in REGIONS.values() for c in chs]

# 1. features_from_data: alpha at 10 Hz in occipital channels, one very noisy channel
n = fs * 60
t = np.arange(n) / fs
data = rng.standard_normal((len(names), n)) * 5e-6
for c in REGIONS["occipital"]:
    data[names.index(c)] += 20e-6 * np.sin(2 * np.pi * 10 * t)
data[names.index("Fz")] += rng.standard_normal(n) * 400e-6  # bad channel
feats, n_rej = features_from_data(data, fs, names)
assert feats.shape[1] == len(FEATURE_NAMES) == 20, feats.shape
occ_alpha = feats[:, FEATURE_NAMES.index("occipital_alpha")].mean()
fro_alpha = feats[:, FEATURE_NAMES.index("frontal_alpha")].mean()
print("epochs kept", len(feats), "rejected", n_rej, "| occipital alpha", round(occ_alpha, 2), "frontal alpha", round(fro_alpha, 2))
assert len(feats) > 0 and n_rej == 0, "one bad channel must not reject epochs"
assert occ_alpha > fro_alpha + 1.0, "occipital alpha should stand out"

# 2. artifact epochs get rejected
bad = data.copy()
bad[:, 20 * fs:24 * fs] += rng.standard_normal((len(names), 4 * fs)) * 3e-3
_, n_rej2 = features_from_data(bad, fs, names)
print("rejected with a 4 s global artifact:", n_rej2)
assert n_rej2 >= 1

# 3. leakage_test on synthetic features: strong subject fingerprints, weak true age effect
rows = []
for s in range(40):
    grp = "old" if s % 2 else "young"
    fingerprint = rng.standard_normal(20) * 1.0
    age_shift = np.zeros(20)
    age_shift[FEATURE_NAMES.index("occipital_alpha")] = -0.3 if grp == "old" else 0.3
    for e in range(40):
        x = fingerprint + age_shift + rng.standard_normal(20) * 0.5
        rows.append({"participant_id": f"sub-{s:03d}", "group": grp, "age": 25, "sex": "F", "epoch": e,
                     **dict(zip(FEATURE_NAMES, x))})
df = pd.DataFrame(rows)
res, _ = leakage_test.run(df, n_repeats=4)
summ = res.groupby(["labels", "split"]).balanced_accuracy.mean()
print(summ.round(3).to_string())
assert summ[("shuffled labels", "epoch")] > 0.7, "epoch split should leak subject identity"
assert abs(summ[("shuffled labels", "subject")] - 0.5) < 0.12, "subject split should sit near chance"
print("synthetic checks passed")
