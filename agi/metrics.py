"""Calibration, structure and stream diagnostics (paper section 2, appendix F.5).

All functions take numpy arrays or torch tensors on CPU.
"""
import numpy as np
import torch


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# ---------------------------------------------------------------- calibration
def ece(conf, correct, n_bins=15):
    """Equal-width-bin ECE (bins (l, u], as in C-TPT's code). 15 bins for every
    number we compute; C-TPT's released code uses 20 (appendix F.5)."""
    conf, correct = _np(conf).astype(float), _np(correct).astype(float)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return total


def gap(conf, correct):
    """Signed calibration gap: mean confidence minus accuracy (ECE >= |gap|)."""
    return float(_np(conf).mean() - _np(correct).mean())


def noise_floor(conf, n_bins=15, n_draws=200, seed=0):
    """Expected ECE of a perfectly calibrated predictor with the same confidences
    on the same test set: labels resampled as Bernoulli(conf)."""
    conf = _np(conf).astype(float)
    rng = np.random.default_rng(seed)
    return float(np.mean([ece(conf, rng.random(len(conf)) < conf, n_bins) for _ in range(n_draws)]))


def debiased_ece(conf, correct, n_bins=15):
    """Debiased calibration error of Kumar et al. (2019): per-bin plug-in squared
    error minus its finite-sample bias, returned as sqrt (an l2 calibration error).
    UNCERTAIN: the paper (table 8, 'deb.') does not state whether it reports the
    l1 or l2 debiased estimate; this is the l2 estimator of Kumar et al."""
    conf, correct = _np(conf).astype(float), _np(correct).astype(float)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        n = m.sum()
        if n > 1:
            a = correct[m].mean()
            total += n / len(conf) * ((a - conf[m].mean()) ** 2 - a * (1 - a) / (n - 1))
    return float(np.sqrt(max(total, 0.0)))


def smooth_ece(conf, correct):
    """Smooth ECE (Blasiok & Nakkiran, 2024) via the `relplot` package, if installed."""
    try:
        import relplot
    except ImportError:
        return None
    return float(relplot.smECE(_np(conf).astype(float), _np(correct).astype(float)))


def class_conditional_ece(conf, correct, labels, classes, n_bins=15):
    """Mean over `classes` of the ECE of the predictions on images of that class
    (split tests, section 5.3). Classes with no image are skipped."""
    conf, correct, labels = _np(conf), _np(correct), _np(labels)
    vals = [ece(conf[labels == k], correct[labels == k], n_bins) for k in classes if (labels == k).any()]
    return float(np.mean(vals)) if vals else float('nan')


def cumulative_ece(conf, correct, n_points=50, n_bins=15):
    """ECE of the first t predictions along the stream (fig. 1b)."""
    conf, correct = _np(conf), _np(correct)
    ts = np.linspace(len(conf) / n_points, len(conf), n_points).astype(int)
    return ts, np.array([ece(conf[:t], correct[:t], n_bins) for t in ts])


def paired_bootstrap(stat_a, stat_b, n_items, n_runs=1, n_resamples=1000, seed=0):
    """95% paired bootstrap interval of mean_r[stat_a(r, idx) - stat_b(r, idx)],
    resampling test images and seeds/orders jointly; both methods share the resamples.
    stat_x(run, idx) -> float for run index `run` restricted to images `idx`.
    A difference is 'resolved' when the interval excludes 0."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        runs = rng.integers(0, n_runs, n_runs)
        idx = rng.integers(0, n_items, n_items)
        diffs.append(np.mean([stat_a(r, idx) - stat_b(r, idx) for r in runs]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), bool(lo > 0 or hi < 0)


# ---------------------------------------------------------------- class-similarity structure
def _ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='stable')] = np.arange(len(x))
    return r


def spearman_structure(T0, T1):
    """Spearman r_s between the N(N-1)/2 pairwise cosines of T1 and of zero-shot T0."""
    T0, T1 = _np(T0).astype(float), _np(T1).astype(float)
    iu = np.triu_indices(T0.shape[0], 1)
    a, b = (T0 @ T0.T)[iu], (T1 @ T1.T)[iu]
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def jaccard_at_k(T0, T1, k=10):
    """Mean over classes of the Jaccard overlap of the k nearest classes before/after
    (k = 10; k = 3 on EuroSAT, appendix F.5)."""
    T0, T1 = _np(T0).astype(float), _np(T1).astype(float)
    out = []
    for T in (T0, T1):
        S = T @ T.T
        np.fill_diagonal(S, -np.inf)
        out.append(np.argsort(-S, axis=1)[:, :k])
    vals = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(*out)]
    return float(np.mean(vals))


# ---------------------------------------------------------------- stream diagnostics
def absorbed_classes(preds, labels, window=5000, frac=0.01):
    """Classes present in the last `window` images of the stream but predicted at
    less than `frac` of their frequency there."""
    preds, labels = _np(preds)[-window:], _np(labels)[-window:]
    n = 0
    for k in np.unique(labels):
        if (preds == k).sum() < frac * (labels == k).sum():
            n += 1
    return n
