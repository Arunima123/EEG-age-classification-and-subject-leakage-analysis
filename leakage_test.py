"""Does splitting epochs at random inflate accuracy? Age-group classification, two ways.

Usage: python leakage_test.py features.csv
Needs: pip install scikit-learn

Four conditions, each repeated with different random splits:
  epoch split,   true labels      random split of epochs (same subject in train and test)
  subject split, true labels      all epochs of a subject stay on one side
  epoch split,   shuffled labels  age labels randomly reassigned between SUBJECTS
  subject split, shuffled labels  same shuffle, subject-wise split
With shuffled labels there is nothing to learn about age, so any accuracy above 0.5 can
only come from recognising individual subjects. That is the leakage.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5
N_REPEATS = 20


def cv_score(X, y, groups, scheme, seed):
    if scheme == "epoch":
        splits = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y)
    else:
        splits = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y, groups)
    scores = []
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores))


def shuffled_labels(subject_ids, subject_labels, rng):
    """Randomly reassign the subject-level labels between subjects, keep class sizes."""
    perm = dict(zip(subject_ids, rng.permutation(subject_labels)))
    return perm


def run(df, n_repeats=N_REPEATS):
    feat_cols = [c for c in df.columns if "_" in c and c.split("_")[1] in ("delta", "theta", "alpha", "beta")]
    X = df[feat_cols].to_numpy()
    groups = df["participant_id"].to_numpy()
    y_true = (df["group"] == "old").astype(int).to_numpy()
    subj = df.groupby("participant_id")["group"].first()
    subj_ids, subj_lab = subj.index.to_numpy(), (subj == "old").astype(int).to_numpy()

    rows = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(1000 + rep)
        mapping = shuffled_labels(subj_ids, subj_lab, rng)
        y_shuf = np.array([mapping[g] for g in groups])
        for cond, y in (("true labels", y_true), ("shuffled labels", y_shuf)):
            for scheme in ("epoch", "subject"):
                rows.append({"repeat": rep, "labels": cond, "split": scheme,
                             "balanced_accuracy": cv_score(X, y, groups, scheme, seed=rep)})
    return pd.DataFrame(rows), feat_cols


def plot(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["true labels", "shuffled labels"]
    w = 0.35
    for i, scheme in enumerate(("epoch", "subject")):
        sub = summary[summary.split == scheme].set_index("labels").loc[labels]
        ax.bar(np.arange(2) + (i - 0.5) * w, sub["mean"], w, yerr=sub["sd"], capsize=3,
               label=f"{scheme}-wise split", color=("tab:red", "tab:blue")[i])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("balanced accuracy (mean ± SD over repeats)")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="upper right")
    ax.set_title("Young vs old from resting EEG: effect of the split")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main(path):
    df = pd.read_csv(path)
    print(f"{len(df)} epochs, {df.participant_id.nunique()} subjects, "
          f"{df.groupby('participant_id').group.first().value_counts().to_dict()}")
    res, feats = run(df)
    print(f"{len(feats)} features")
    res.to_csv("leakage_results_raw.csv", index=False)
    summary = (res.groupby(["labels", "split"]).balanced_accuracy
               .agg(mean="mean", sd="std").reset_index())
    print(summary.round(3).to_string(index=False))
    summary.to_csv("leakage_results.csv", index=False)
    plot(summary, "leakage_figure.png")
    print("saved leakage_results.csv, leakage_results_raw.csv, leakage_figure.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "features.csv")
