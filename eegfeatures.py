"""Epoching and band-power features from a filtered EEG array. Pure numpy/scipy, no MNE.

Design choices, all meant to survive a few bad channels:
- median reference across channels (an average reference would spread one bad channel
  into every other channel)
- 4 s epochs, an epoch is dropped when its median channel peak-to-peak exceeds a threshold
- one value per brain region and band: the MEDIAN across channels in the region of the
  log band power, so a single noisy electrode does not change the feature
"""
import numpy as np
from scipy import signal

REGIONS = {
    "frontal": ["Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
    "central": ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
    "temporal": ["FT9", "FT7", "FT8", "FT10", "T7", "T8", "TP7", "TP8", "TP9", "TP10"],
    "parietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
    "occipital": ["PO9", "PO7", "PO3", "POz", "PO4", "PO8", "PO10", "O1", "Oz", "O2"],
}
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0)}
FEATURE_NAMES = [f"{r}_{b}" for b in BANDS for r in REGIONS]


def features_from_data(data, fs, ch_names, epoch_sec=4.0, edge_sec=5.0, reject_uv=250.0):
    """data: channels x samples in volts, already band-pass filtered.
    Returns (features: epochs x 20, n_rejected_epochs)."""
    ch_names = list(ch_names)
    data = data - np.median(data, axis=0, keepdims=True)
    e = int(edge_sec * fs)
    data = data[:, e:data.shape[1] - e]
    n = int(epoch_sec * fs)
    n_ep = data.shape[1] // n
    ep = data[:, : n_ep * n].reshape(len(ch_names), n_ep, n).transpose(1, 0, 2)  # epochs x ch x samples

    p2p_uv = np.ptp(ep, axis=2) * 1e6
    keep = np.median(p2p_uv, axis=1) < reject_uv
    ep = ep[keep]
    if len(ep) == 0:
        return np.empty((0, len(FEATURE_NAMES))), int((~keep).sum())

    nper = int(2 * fs)
    f, psd = signal.welch(ep, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    df = f[1] - f[0]
    region_idx = {r: [ch_names.index(c) for c in chs] for r, chs in REGIONS.items()}
    cols = []
    for band, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f <= hi)
        bp = psd[:, :, m].sum(axis=-1) * df  # epochs x channels
        for r, idx in region_idx.items():
            cols.append(np.log10(np.median(bp[:, idx], axis=1)))
    return np.stack(cols, axis=1), int((~keep).sum())
