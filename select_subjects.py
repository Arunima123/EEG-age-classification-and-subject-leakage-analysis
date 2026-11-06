"""Pick a sex-balanced young vs old subject list from ds005385 participants.tsv.

Usage:  python select_subjects.py ds005385/participants.tsv --per-cell 10
Writes subjects.csv (participant_id, sex, age, group) and prints the subject IDs.

Rules (from the dataset README and participants.json):
- young = age 20-30, old = age 60-70
- late_ses1 must be 0: it counts late triggers, and nonzero values mean the recording
  is probably not continuous
- equal numbers of F and M in each age group, because the raw groups differ a lot
  in sex ratio (young is mostly female), and a classifier could otherwise learn sex
"""
import argparse

import pandas as pd


def select(path, per_cell, seed):
    d = pd.read_csv(path, sep="\t")
    d["late_ses1"] = pd.to_numeric(d["late_ses1"], errors="coerce")
    d = d[(d.session1 == "yes") & (d.late_ses1 == 0)].copy()
    d["group"] = None
    d.loc[d.age <= 30, "group"] = "young"
    d.loc[d.age >= 60, "group"] = "old"
    d = d.dropna(subset=["group"])
    parts = []
    for (g, s), cell in d.groupby(["group", "sex"]):
        if len(cell) < per_cell:
            raise SystemExit(f"only {len(cell)} subjects in {g}/{s}, need {per_cell}")
        parts.append(cell.sample(per_cell, random_state=seed))
    out = pd.concat(parts).sort_values("participant_id")
    return out[["participant_id", "sex", "age", "group"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("participants")
    ap.add_argument("--per-cell", type=int, default=10, help="subjects per group x sex cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="subjects.csv")
    a = ap.parse_args()
    sel = select(a.participants, a.per_cell, a.seed)
    sel.to_csv(a.out, index=False)
    print(sel.groupby(["group", "sex"]).age.agg(["count", "mean"]).round(1))
    print("subjects:", " ".join(sel.participant_id))
    print("saved", a.out)
