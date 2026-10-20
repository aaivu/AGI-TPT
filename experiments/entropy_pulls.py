"""Entropy minimization pulls competitors together (proposition 3, fig. 3a, section 5.3).

For each image: take ONE small entropy step and record, for every class pair, whether
its cosine s_ij rose. Classes are 'competitive' when p_k > exp(-H(p)) (p computed at
the start of the step). Pairs are grouped as both competitive / mixed / neither, and
we report the share of pairs whose cosine rises in each group, for
  * free class features: T itself is the parameter (renormalized after the step),
  * the 4-token prompt: the context pi is the parameter.
Prediction (prop. 3): high share for both-competitive, low share for neither.

Usage (run by the user):
  python experiments/entropy_pulls.py DATA --test_sets I --n_images 1000 --out results/entropy_pulls_I.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agi_tpt_classification import get_dataset, tpt_transform, avg_entropy, select_confident_samples  # noqa: E402
from clip.custom_clip import get_coop  # noqa: E402
from utils.tools import set_random_seed  # noqa: E402


def groups(p, H):
    comp = p > torch.exp(-H)
    both = comp[:, None] & comp[None, :]
    neither = ~comp[:, None] & ~comp[None, :]
    mixed = ~both & ~neither
    triu = torch.ones_like(both).triu(1)
    return {'both': both & triu, 'mixed': mixed & triu, 'neither': neither & triu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--test_sets', default='I')
    ap.add_argument('--arch', default='ViT-B/16')
    ap.add_argument('--n_images', type=int, default=1000)
    ap.add_argument('--lr_free', type=float, default=1e-3, help='step size on free features')
    ap.add_argument('--lr_prompt', type=float, default=1e-4, help='SGD step size on the prompt')
    ap.add_argument('--out', default='results/entropy_pulls.json')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    a.resolution, a.batch_size, a.I_augmix, a.dataset_mode = 224, 64, False, 'test'
    a.classnames_file, a.selection_p = None, 0.1

    set_random_seed(0)
    device = torch.device(f'cuda:{a.gpu}')
    ds, classnames = get_dataset(a, tpt_transform(a))
    model = get_coop(a.arch, classnames, device, 4, 'a_photo_of_a')
    for n, q in model.named_parameters():
        q.requires_grad_(n == 'prompt_learner.ctx')
    ctx = model.prompt_learner.ctx
    with torch.no_grad():
        T0 = model.get_text_features()

    counts = {s: {g: [0, 0] for g in ('both', 'mixed', 'neither')} for s in ('free', 'prompt')}
    idx = np.random.default_rng(0).choice(len(ds), size=min(a.n_images, len(ds)), replace=False)
    for i in idx:
        views, _ = ds[int(i)]
        img = model.encode_image(torch.stack(views).to(device))
        with torch.no_grad():
            logits = model.logits(img, T0)
            sel = select_confident_samples(logits, a.selection_p)
            p = logits[sel].softmax(1).mean(0)
            H = -(p * p.clamp_min(1e-30).log()).sum()
        G = groups(p, H)
        S0 = T0 @ T0.t()

        # free features
        T = T0.clone().requires_grad_(True)
        g, = torch.autograd.grad(avg_entropy(model.logits(img, T)[sel]), T)
        with torch.no_grad():
            T1 = T0 - a.lr_free * g
            T1 = T1 / T1.norm(dim=1, keepdim=True)
            rise = (T1 @ T1.t()) > S0
        for k, m in G.items():
            counts['free'][k][0] += int((rise & m).sum()); counts['free'][k][1] += int(m.sum())

        # 4-token prompt
        model.reset()
        g, = torch.autograd.grad(avg_entropy(model.logits(img, model.get_text_features())[sel]), ctx)
        with torch.no_grad():
            ctx.sub_(a.lr_prompt * g / g.abs().max().clamp_min(1e-30))
            T1 = model.get_text_features()
            rise = (T1 @ T1.t()) > S0
            model.reset()
        for k, m in G.items():
            counts['prompt'][k][0] += int((rise & m).sum()); counts['prompt'][k][1] += int(m.sum())

    out = {s: {k: (v[0] / v[1] if v[1] else None) for k, v in d.items()} for s, d in counts.items()}
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'share_rising': out, 'counts': counts}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
