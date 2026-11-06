"""Load one ds005385 recording and check that it looks sane.

Usage: python check_one.py
       python check_one.py path/to/other_file.edf

Prints channel names, sampling rate, duration and typical amplitude, and saves
check_psd.png (power spectrum). The README warns that physical min/max values in the
EDF headers may be invalid, so the amplitude line is the important check.
"""
import sys

import numpy as np
import mne

DEFAULT = "ds005385/sub-001/ses-1/eeg/sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
print("file:", path)
print("sampling rate (Hz):", raw.info["sfreq"])
print("duration (s):", round(raw.times[-1], 1))
print("n channels:", len(raw.ch_names))
print("channel names:", raw.ch_names)
print("channel types:", sorted(set(raw.get_channel_types())))

eeg = raw.copy().pick("eeg")
data = eeg.get_data()
std_uv = np.std(data, axis=1) * 1e6
print(f"EEG channels: {len(eeg.ch_names)}")
print(f"median channel std (microvolts): {np.median(std_uv):.1f}  (min {std_uv.min():.1f}, max {std_uv.max():.1f})")
print("Expected: tens of microvolts. Values near 0.001 or in the thousands mean the EDF scaling is wrong.")

eeg.filter(1.0, 40.0, verbose="ERROR")
psd = eeg.compute_psd(fmin=1.0, fmax=40.0, n_fft=int(2 * eeg.info["sfreq"]), verbose="ERROR")
fig = psd.plot(show=False)
fig.savefig("check_psd.png", dpi=150)
print("saved check_psd.png (look for a bump between 8 and 12 Hz)")
