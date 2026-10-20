"""Equal strength at one step (section 5.1, appendix F.5, table 7).

Measures the override mass mu = E_x ||g_x|_{O_x}||_1 / ||g_x||_1 of a regularizer at
pi_0, without labels, on `--n_images` images of the training split, and rescales its
weight (bisection on log-weight; mu is non-decreasing along the ray of weights)
until mu equals AGI-TPT's mu at lambda = 4 on the same images.

Because b_R = lambda * grad R(T(pi_0)) is the same for every image (proposition 1),
grad R is computed once; only the TPT gradient g_x is computed per image.

Output: JSON {method: {"weight": w, "mu": mu, "nu": nu}} to pass as --lam (and,
for the random flip set, AGI-TPT's mu as --flip_mu).

Usage (run by the user):
  python experiments/equal_strength.py DATA --test_sets I --methods ctpt otpt dtpt atpt \
      --init_weights 50 18 1e5 80 --out results/equal_strength_I.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agi_tpt_classification import get_dataset, tpt_transform, avg_entropy, select_confident_samples, grad_of  # noqa: E402
from agi import Regularizer  # noqa: E402
from clip.custom_clip import get_coop  # noqa: E402
from utils.tools import set_random_seed  # noqa: E402


def mean_override(gs, b):
    """mu and nu averaged over images for the fixed offset b."""
    mus, nus = [], []
    for g in gs:
        O = (torch.sign(g + b) != torch.sign(g)) & (g != 0)
        mus.append(((g.abs() * O).sum() / g.abs().sum()).item())
        nus.append(O.float().mean().item())
    return float(np.mean(mus)), float(np.mean(nus))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data')
    p.add_argument('--test_sets', default='I')
    p.add_argument('--arch', default='ViT-B/16')
    p.add_argument('--n_images', type=int, default=500)
    p.add_argument('--methods', nargs='+', default=['ctpt', 'otpt', 'dtpt', 'atpt'])
    p.add_argument('--init_weights', nargs='+', type=float, default=[50, 18, 1e5, 80],
                   help='published weights, starting points of the search')
    p.add_argument('--agi_lam', type=float, default=4.0)
    p.add_argument('--out', default='results/equal_strength.json')
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()

    # defaults expected by the helpers of the main script
    a.resolution, a.batch_size, a.I_augmix, a.dataset_mode = 224, 64, False, 'train'
    a.classnames_file, a.selection_p, a.loss_scale = None, 0.1, 1000.0
    set_random_seed(0)
    device = torch.device(f'cuda:{a.gpu}')
    ds, classnames = get_dataset(a, tpt_transform(a), mode='train')
    model = get_coop(a.arch, classnames, device, 4, 'a_photo_of_a')
    for n, q in model.named_parameters():
        q.requires_grad_(n == 'prompt_learner.ctx')
    ctx = model.prompt_learner.ctx

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(a.n_images, len(ds)), replace=False)
    gs = []
    for i in idx:
        views, _ = ds[int(i)]
        views = torch.stack(views).to(device)
        img = model.encode_image(views)
        T = model.get_text_features()
        logits = model.logits(img, T)
        L = avg_entropy(logits[select_confident_samples(logits.detach(), a.selection_p)])
        gs.append(grad_of(L, ctx, a.loss_scale, retain=False).detach())

    def reg_grad(name):
        T = model.get_text_features()
        R = Regularizer(name, T.shape[0], T.shape[1], device=device)(T)
        return grad_of(R, ctx, a.loss_scale, retain=False).detach()

    g_agi = reg_grad('agi')
    mu_t, nu_t = mean_override(gs, a.agi_lam * g_agi)
    out = {'agi': {'weight': a.agi_lam, 'mu': mu_t, 'nu': nu_t}}
    print(f"AGI-TPT (lambda={a.agi_lam}): mu={mu_t:.3f} nu={nu_t:.3f}")

    for name, w0 in zip(a.methods, a.init_weights):
        gr = reg_grad(name)
        mu0, nu0 = mean_override(gs, w0 * gr)
        lo, hi = math.log(w0) - 12, math.log(w0) + 12
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mean_override(gs, math.exp(mid) * gr)[0] < mu_t:
                lo = mid
            else:
                hi = mid
        w = math.exp(0.5 * (lo + hi))
        mu, nu = mean_override(gs, w * gr)
        out[name] = {'weight': w, 'mu': mu, 'nu': nu, 'published_weight': w0,
                     'mu_published': mu0, 'nu_published': nu0}
        print(f"{name}: published w={w0:g} (mu={mu0:.3f}, nu={nu0:.3f}) -> matched w={w:.4g} (mu={mu:.3f})")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
