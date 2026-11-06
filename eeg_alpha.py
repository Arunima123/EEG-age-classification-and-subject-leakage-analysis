"""Eyes open vs eyes closed alpha power (the Berger effect).

Data: PhysioNet EEG Motor Movement/Imagery dataset (109 subjects, 64 channels,
160 Hz). Run 1 = 1 minute eyes open, run 2 = 1 minute eyes closed. MNE downloads
the files on first use (about 2.5 MB per subject for these two runs).

Setup:   pip install mne scipy pandas matplotlib
Run:     python eeg_alpha.py --subjects 1-20 --out results

Question: is alpha-band (8-12 Hz) power over occipital channels higher with
eyes closed than with eyes open, and is the effect specific to the back of the
head (compared with a frontal control region)?
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

ALPHA = (8.0, 12.0)
OCCIPITAL = ["O1", "Oz", "O2"]
FRONTAL = ["Fz", "F3", "F4"]  # control region
EDGE_SEC = 2.0  # trimmed from both ends of each recording after filtering


# ---------- core analysis: plain numpy/scipy, no MNE needed ----------

def welch_psd(data, sfreq):
    """data: channels x samples (volts). Returns freqs, PSD (V^2/Hz)."""
    nper = int(2 * sfreq)  # 2 s windows, 0.5 Hz resolution
    return signal.welch(data, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)


def band_power(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    return psd[:, m].sum(axis=-1) * (freqs[1] - freqs[0])


def analyze(records):
    """records: list of dicts with keys subject, open, closed (channels x samples),
    sfreq, ch_names. Returns (per-subject DataFrame, group arrays)."""
    rows, ch_ratio, psd_occ_open, psd_occ_closed = [], [], [], []
    for r in records:
        names = list(r["ch_names"])
        occ = [names.index(c) for c in OCCIPITAL]
        fro = [names.index(c) for c in FRONTAL]
        f, p_open = welch_psd(r["open"], r["sfreq"])
        _, p_closed = welch_psd(r["closed"], r["sfreq"])
        a_open = band_power(f, p_open, ALPHA)
        a_closed = band_power(f, p_closed, ALPHA)
        rows.append({
            "subject": r["subject"],
            "occ_alpha_open": a_open[occ].mean(),
            "occ_alpha_closed": a_closed[occ].mean(),
            "occ_log10_ratio": np.log10(a_closed[occ].mean() / a_open[occ].mean()),
            "front_alpha_open": a_open[fro].mean(),
            "front_alpha_closed": a_closed[fro].mean(),
            "front_log10_ratio": np.log10(a_closed[fro].mean() / a_open[fro].mean()),
        })
        ch_ratio.append(np.log10(a_closed / a_open))
        psd_occ_open.append(np.log10(p_open[occ].mean(axis=0)))
        psd_occ_closed.append(np.log10(p_closed[occ].mean(axis=0)))
    df = pd.DataFrame(rows)
    group = {
        "freqs": f,
        "ch_log_ratio": np.mean(ch_ratio, axis=0),
        "psd_open": np.array(psd_occ_open),
        "psd_closed": np.array(psd_occ_closed),
    }
    return df, group


def summarize(df):
    """Paired tests on log10(closed/open). H0: log ratio = 0."""
    out = {}
    for roi in ("occ", "front"):
        x = df[f"{roi}_log10_ratio"].to_numpy()
        w = stats.wilcoxon(x)
        t = stats.ttest_1samp(x, 0.0)
        out[roi] = {
            "n": len(x),
            "n_closed_higher": int((x > 0).sum()),
            "mean_ratio_closed_over_open": float(10 ** x.mean()),
            "cohens_dz": float(x.mean() / x.std(ddof=1)),
            "wilcoxon_p": float(w.pvalue),
            "t_p": float(t.pvalue),
        }
    return out


def print_summary(res):
    for roi, label in (("occ", "Occipital (O1, Oz, O2)"), ("front", "Frontal control (Fz, F3, F4)")):
        s = res[roi]
        print(f"{label}: closed > open in {s['n_closed_higher']}/{s['n']} subjects, "
              f"mean ratio {s['mean_ratio_closed_over_open']:.2f}x, "
              f"dz = {s['cohens_dz']:.2f}, Wilcoxon p = {s['wilcoxon_p']:.2g}, t-test p = {s['t_p']:.2g}")


def make_figure(df, group, info=None, path="alpha_figure.png"):
    ncols = 3 if info is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    f = group["freqs"]
    keep = (f >= 1) & (f <= 30)

    ax = axes[0]
    for key, color, lab in (("psd_open", "tab:blue", "Eyes open"), ("psd_closed", "tab:red", "Eyes closed")):
        m = group[key][:, keep]
        ax.plot(f[keep], m.mean(axis=0), color=color, label=lab)
        se = m.std(axis=0, ddof=1) / np.sqrt(m.shape[0])
        ax.fill_between(f[keep], m.mean(axis=0) - se, m.mean(axis=0) + se, color=color, alpha=0.2)
    ax.axvspan(*ALPHA, color="grey", alpha=0.15)
    ax.set(xlabel="Frequency (Hz)", ylabel="log10 PSD (V$^2$/Hz)", title="Occipital PSD (mean ± SEM)")
    ax.legend()

    ax = axes[1]
    for _, r in df.iterrows():
        ax.plot([0, 1], [np.log10(r.occ_alpha_open), np.log10(r.occ_alpha_closed)], color="grey", alpha=0.6, marker="o", ms=3)
    ax.set_xticks([0, 1], ["Eyes open", "Eyes closed"])
    ax.set(ylabel="log10 occipital alpha power (8-12 Hz)", title=f"Per subject (n = {len(df)})")

    if info is not None:
        import mne
        ax = axes[2]
        im, _ = mne.viz.plot_topomap(group["ch_log_ratio"], info, axes=ax, show=False, cmap="RdBu_r")
        ax.set_title("Mean log10(closed / open), alpha")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- data loading: needs MNE ----------

def load_subject(subject):
    import mne
    from mne.datasets import eegbci

    paths = eegbci.load_data(subject, runs=[1, 2])  # downloads to ~/mne_data if missing
    arrays, info = {}, None
    for key, p in zip(("open", "closed"), paths):
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        eegbci.standardize(raw)  # "Oz.." -> "Oz" etc.
        raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
        raw.filter(1.0, 40.0, verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        raw.crop(tmin=EDGE_SEC, tmax=raw.times[-1] - EDGE_SEC)
        arrays[key] = raw.get_data()
        info = raw.info
    return {"subject": subject, "open": arrays["open"], "closed": arrays["closed"],
            "sfreq": info["sfreq"], "ch_names": info["ch_names"]}, info


def parse_subjects(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-20")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records, info = [], None
    for s in parse_subjects(args.subjects):
        print(f"loading subject {s}")
        rec, info = load_subject(s)
        records.append(rec)

    df, group = analyze(records)
    df.to_csv(os.path.join(args.out, "per_subject.csv"), index=False)
    res = summarize(df)
    print_summary(res)
    make_figure(df, group, info, os.path.join(args.out, "alpha_figure.png"))
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
