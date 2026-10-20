"""Aggregate saved runs into the numbers reported in the paper.

Each run of agi_tpt_classification.py writes <tag>.json (metrics) and <tag>.npz
(per-image confidence, prediction, label, stream order). This script:

  summary  mean +- s.d. over seeds (episodic) or stream orders (online) of every
           metric, grouped by dataset / method / setting / lambda / level.
  compare  95% paired bootstrap interval of the ECE (or accuracy) difference between
           two methods over images and seeds/orders (1000 resamples); a difference is
           'resolved' when the interval excludes 0 (appendix F.5).
  curve    cumulative ECE along the stream (fig. 1b).

Usage:
  python analysis/aggregate.py summary results/ --out results/summary.csv
  python analysis/aggregate.py compare results/ --a "I_ViT-B16_agi_online" --b "I_ViT-B16_atpt_online"
  python analysis/aggregate.py curve results/I_ViT-B16_agi_online_lam4.0_seed0_order0.npz
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agi.metrics import ece, paired_bootstrap, cumulative_ece  # noqa: E402

GROUP_KEYS = ('test_sets', 'arch', 'reg', 'setting', 'lam', 'level_mode', 'level_offset',
              'anneal', 'remedy', 'control', 'split_kind', 'tta_steps')


def load_runs(folder):
    runs = []
    for f in sorted(glob.glob(os.path.join(folder, '*.json'))):
        r = json.load(open(f))
        if 'config' in r:
            r['_path'] = f[:-5]
            runs.append(r)
    return runs


def summary(a):
    groups = defaultdict(list)
    for r in load_runs(a.folder):
        groups[tuple(r['config'].get(k) for k in GROUP_KEYS)].append(r)
    metrics = ['acc', 'ece', 'ece20', 'gap', 'noise_floor', 'debiased_ece', 'smooth_ece', 'ece_end',
               'spearman_rs', 'jaccard', 'absorbed_classes', 'force_below_cstar_pct', 'reached_pct',
               'ece_changed_half', 'ece_other_half', 'override_mass_mu', 'override_fraction_nu']
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(GROUP_KEYS, key))
        row['n_runs'] = len(rs)
        for m in metrics:
            v = [r[m] for r in rs if r.get(m) is not None]
            if v:
                row[m] = round(float(np.mean(v)), 3)
                row[m + '_sd'] = round(float(np.std(v)), 3)
        rows.append(row)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in GROUP_KEYS, k))
    with open(a.out, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


def _load_npz(folder, prefix):
    files = sorted(glob.glob(os.path.join(folder, prefix + '*.npz')))
    assert files, f"no runs match {prefix}"
    out = []
    for f in files:
        d = np.load(f)
        # align images across methods by dataset index (stream orders differ)
        o = np.argsort(d['order'])
        out.append((d['conf'][o], (d['pred'] == d['label'])[o]))
    return out


def compare(a):
    A, B = _load_npz(a.folder, a.a), _load_npz(a.folder, a.b)
    n_runs = min(len(A), len(B))
    n = len(A[0][0])

    def stat(runs):
        if a.metric == 'acc':
            return lambda r, idx: 100 * runs[r][1][idx].mean()
        return lambda r, idx: 100 * ece(runs[r][0][idx], runs[r][1][idx], a.bins)

    lo, hi, resolved = paired_bootstrap(stat(A), stat(B), n, n_runs, a.resamples)
    print(f"{a.metric}: {a.a} - {a.b}: 95% CI [{lo:.3f}, {hi:.3f}]  resolved={resolved}")


def curve(a):
    d = np.load(a.npz)
    ts, e = cumulative_ece(d['conf'], d['pred'] == d['label'], a.points, a.bins)
    for t, v in zip(ts, e):
        print(f"{t}\t{100 * v:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summary'); s.add_argument('folder'); s.add_argument('--out', default='summary.csv')
    c = sub.add_parser('compare'); c.add_argument('folder'); c.add_argument('--a', required=True)
    c.add_argument('--b', required=True); c.add_argument('--metric', choices=['ece', 'acc'], default='ece')
    c.add_argument('--bins', type=int, default=15); c.add_argument('--resamples', type=int, default=1000)
    k = sub.add_parser('curve'); k.add_argument('npz'); k.add_argument('--points', type=int, default=50)
    k.add_argument('--bins', type=int, default=15)
    a = p.parse_args()
    {'summary': summary, 'compare': compare, 'curve': curve}[a.cmd](a)
