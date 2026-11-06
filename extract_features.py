"""Turn each downloaded eyes-closed recording into per-epoch band-power features.

Usage: python extract_features.py subjects.csv
Writes features.csv with one row per 4 s epoch.
Needs MNE (installed already). Takes a few minutes for 40 subjects.
"""
import os
import sys

import mne
import pandas as pd

from eegfeatures import FEATURE_NAMES, REGIONS, features_from_data

FS = 250  # resample target (Hz)


def path_for(sid):
    return f"ds005385/{sid}/ses-1/eeg/{sid}_ses-1_task-EyesClosed_acq-pre_eeg.edf"


def load(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick("eeg")  # drops the Status trigger channel
    raw.filter(1.0, 40.0, verbose="ERROR")
    raw.resample(FS, verbose="ERROR")
    return raw.get_data(), raw.ch_names


def main(subjects_csv):
    subs = pd.read_csv(subjects_csv)
    needed = {c for chs in REGIONS.values() for c in chs}
    frames = []
    for i, row in enumerate(subs.itertuples(), 1):
        p = path_for(row.participant_id)
        if not os.path.exists(p):
            print(f"[{i}/{len(subs)}] {row.participant_id}: file missing, skipped")
            continue
        try:
            data, names = load(p)
        except Exception as e:  # truncated or corrupt download
            print(f"[{i}/{len(subs)}] {row.participant_id}: could not read file ({type(e).__name__}), skipped. "
                  "Re-run download_subjects.py, then this script.")
            continue
        missing = needed - set(names)
        if missing:
            print(f"{row.participant_id}: missing channels {sorted(missing)}, skipped")
            continue
        feats, n_rej = features_from_data(data, FS, names)
        print(f"[{i}/{len(subs)}] {row.participant_id}: {len(feats)} epochs kept, {n_rej} rejected")
        if len(feats) == 0:
            continue
        df = pd.DataFrame(feats, columns=FEATURE_NAMES)
        df.insert(0, "epoch", range(len(df)))
        for col in ("group", "age", "sex", "participant_id"):
            df.insert(0, col, getattr(row, col))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv("features.csv", index=False)
    print(f"saved features.csv: {len(out)} epochs from {out.participant_id.nunique()} subjects")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "subjects.csv")
