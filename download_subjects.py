"""Download only the session-1, eyes-closed, pre-task recording for the subjects in subjects.csv.

Usage: python download_subjects.py subjects.csv
Each recording is roughly 23 MB.

Network errors happen. This script retries each subject a few times, carries on with the
next subject if one keeps failing, and lists the failures at the end. Re-running it is safe:
openneuro-py checks every file and only downloads what is missing or incomplete.
"""
import sys
import time

import pandas as pd
import openneuro as on

DATASET = "ds005385"
TARGET = "ds005385"
ATTEMPTS = 4


def files_for(sid):
    base = f"{sid}/ses-1/eeg/{sid}_ses-1_task-EyesClosed_acq-pre"
    return [f"{base}_eeg.edf", f"{base}_eeg.json", f"{base}_channels.tsv", f"{base}_events.tsv"]


def download_one(sid):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            on.download(dataset=DATASET, target_dir=TARGET, include=files_for(sid))
            return True
        except Exception as e:  # openneuro-py raises RuntimeError on network failures
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            print(f"  attempt {attempt}/{ATTEMPTS} failed for {sid}: {first_line}")
            time.sleep(5 * attempt)
    return False


def main(path):
    subjects = pd.read_csv(path)["participant_id"].tolist()
    failed = []
    for i, sid in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {sid}")
        if not download_one(sid):
            failed.append(sid)
    if failed:
        print(f"\n{len(failed)} subject(s) failed: {' '.join(failed)}")
        print("Run the same command again to retry them. Files that are already complete are skipped.")
    else:
        print("\nall subjects downloaded")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "subjects.csv")
