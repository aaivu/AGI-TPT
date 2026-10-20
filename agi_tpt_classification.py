"""AGI-TPT: Adaptive Geometry-Informed Test-Time Prompt Tuning.

One script for every run in the paper (section 5, appendices F-G):

  settings   --setting episodic   one TPT step per image, then reset (5 seeds)
             --setting episodic --tta_steps k
             --setting online     prompt (and AdamW state) carried along the stream,
                                  weight decay 0, 10 stream orders (--stream_seed 0..9)
  methods    --reg none (TPT) | agi (AGI-TPT) | agi_fixed | agi_l0 | agi_lsimplex |
             agi_nolevel | agi_angle | agi_nn | agi_perclass | ctpt | otpt | dtpt |
             atpt | etf_nc | split          (--no_tune: zero-shot CLIP)
  levels     --level_mode / --level_offset (xi) / --level_value      (section 5.4, G.7, G.8)
  controls   --control random_flip --flip_mu MU                       (table 7)
             --anneal offset_norm|push_mass --trace_in RUN_trace.json (table 11)
             --anneal per_pair                                         (table 11)
  remedies   --remedy periodic_reset | ema | sar                       (table 9)
  split test --reg split --split_kind two_sided|level0                 (fig. 3c)

AGI-TPT for one image (algorithm 1):
  c <- c*(N, D);  T <- normalize(f_T(pi, classes));  S <- T T^T
  L_AGI = 1/N^2 ||ReLU(S - G_c)||_F^2;  L_TPT = entropy of the 10% most confident views
  pi <- AdamW(pi, grad(L_TPT + lambda L_AGI));  predict.

Nothing in this file is executed at import time.
"""
import argparse
import json
import math
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, build_extra_dataset, EXTRA_SETS
from data.cls_to_names import *  # noqa: F401,F403  (<set>_classes lists)
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask  # noqa: F401
from utils.tools import set_random_seed
from agi import Regularizer, metrics
from agi.strength import override_stats, random_flip_grad

NORMALIZE = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])


# ============================================================== data
def get_dataset(args, transform, mode=None):
    """Returns (dataset, classnames)."""
    set_id, mode = args.test_sets, mode or args.dataset_mode
    if set_id in EXTRA_SETS:
        return build_extra_dataset(set_id, transform, args.data, mode, args.classnames_file)
    if len(set_id) > 1:
        classnames = eval("{}_classes".format(set_id.lower()))
    else:
        assert set_id in ['A', 'R', 'K', 'V', 'I']
        if set_id == 'R':
            classnames = [c for c, m in zip(imagenet_classes, imagenet_r_mask) if m]
        elif set_id in ['A', 'V']:
            classnames = [imagenet_classes[i] for i in eval("imagenet_{}_mask".format(set_id.lower()))]
        else:
            classnames = imagenet_classes
    return build_dataset(set_id, transform, args.data, mode=mode), classnames


def tpt_transform(args):
    base = transforms.Compose([transforms.Resize(args.resolution, interpolation=BICUBIC),
                               transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([transforms.ToTensor(), NORMALIZE])
    # AugMix for the fine-grained sets as in TPT (and ImageNet with --I_augmix)
    augmix = len(args.test_sets) > 1 or args.I_augmix
    return AugMixAugmenter(base, preprocess, n_views=args.batch_size - 1, augmix=augmix)


def plain_transform(args):
    return transforms.Compose([transforms.Resize(args.resolution, interpolation=BICUBIC),
                               transforms.CenterCrop(args.resolution), transforms.ToTensor(), NORMALIZE])


def dataset_labels(ds):
    for attr in ('targets', 'labels', 'label_list'):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    if hasattr(ds, '_samples'):                       # torchvision FER2013
        return np.asarray([s[1] for s in ds._samples])
    return np.asarray([int(ds[i][1]) for i in range(len(ds))])


def stream_order(args, ds):
    """Indices of the test stream. Seeded random permutation per pass (stream orders
    0-9 shared by all methods). --label_shift a: temporally correlated labels, each
    class's images spread over time by a Dirichlet(a) draw (smaller a = stronger shift).
    UNCERTAIN: the exact label-shift protocol of Supp. S6.7 was not available; this is
    the common Dirichlet ordering used in practical test-time adaptation."""
    n = len(ds)
    order = []
    for p in range(args.passes):
        rng = np.random.default_rng(args.stream_seed + 1000 * p)
        if args.label_shift is None:
            order.extend(rng.permutation(n).tolist())
        else:
            labels = dataset_labels(ds)
            n_slots = 100
            slots = [[] for _ in range(n_slots)]
            for k in np.unique(labels):
                idx = rng.permutation(np.where(labels == k)[0])
                share = rng.dirichlet([args.label_shift] * n_slots)
                cuts = (np.cumsum(share) * len(idx)).astype(int)[:-1]
                for s, part in enumerate(np.split(idx, cuts)):
                    slots[s].extend(part.tolist())
            for s in slots:
                order.extend(rng.permutation(s).tolist())
    if args.max_images:
        order = order[:args.max_images]
    return order


# ============================================================== TPT pieces
def select_confident_samples(logits, top):
    ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(ent, descending=False)[:int(ent.size(0) * top)]
    return idx


def avg_entropy(outputs):
    """Entropy of the prediction averaged over views (TPT)."""
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg = logits.logsumexp(dim=0) - math.log(logits.shape[0])
    avg = torch.clamp(avg, min=torch.finfo(avg.dtype).min)
    return -(avg * torch.exp(avg)).sum(dim=-1)


def grad_of(loss, ctx, scale, retain):
    """d loss / d ctx with a static loss scale (fp16 text encoder)."""
    g, = torch.autograd.grad(loss * scale, ctx, retain_graph=retain)
    return g / scale


# ============================================================== one test image
class Adapter:
    """Holds the prompt, optimizer, regularizer and stream state for one run."""

    def __init__(self, args, model, N, D, device):
        self.args, self.model, self.device = args, model, device
        self.ctx = model.prompt_learner.ctx
        wd = args.wd if args.wd is not None else (0.0 if args.setting == 'online' else 0.01)
        self.optimizer = torch.optim.AdamW([self.ctx], lr=args.lr, weight_decay=wd)
        self.optim_state = deepcopy(self.optimizer.state_dict())
        self.reg = None
        if args.reg != 'none':
            self.reg = Regularizer(args.reg, N, D, args.level_mode, args.level_offset, args.level_value,
                                   args.split_kind, args.split_seed, args.anneal == 'per_pair', device)
        self.c_star = Regularizer('agi', N, D, device=device).c_star if N > 1 else -1.0
        self.ema_ctx = self.ctx.detach().clone() if args.remedy == 'ema' else None
        self.sar_ema = None
        self.target = json.load(open(args.trace_in)) if args.trace_in else None
        self.trace = {'offset_norm': [], 'pair_force': [], 'mu': [], 'nu': [], 'entropy': []}
        # online AGI-TPT runs always keep the trace the annealed controls match
        self.save_trace = args.trace_out is not None or (
            args.reg == 'agi' and args.setting == 'online' and args.anneal == 'none')
        self.flip_gen = torch.Generator().manual_seed(args.seed)
        # diagnostics (appendix F.5)
        self.force_below, self.force_total = 0.0, 0.0
        self.reached = torch.zeros(N, N, dtype=torch.bool, device=device) if args.diagnostics else None
        self.t = 0

    def reset(self):
        with torch.no_grad():
            self.model.reset()
        self.optimizer.load_state_dict(self.optim_state)

    def weight(self, g_reg, F_pair):
        """lambda, or the annealed weight that matches a recorded AGI-TPT trace."""
        a, lam = self.args, self.args.lam
        if a.anneal == 'offset_norm':
            tgt = self.target['offset_norm']
            return tgt[min(self.t, len(tgt) - 1)] / g_reg.norm().clamp_min(1e-30).item()
        if a.anneal == 'push_mass':
            tgt = self.target['pair_force']
            return tgt[min(self.t, len(tgt) - 1)] / F_pair.abs().sum().clamp_min(1e-30).item()
        return lam

    def step(self, img_feats, sel_idx):
        a, model, ctx = self.args, self.model, self.ctx
        T = model.get_text_features()
        logits = model.logits(img_feats, T)
        if sel_idx is None:
            sel_idx = select_confident_samples(logits.detach(), a.selection_p)
        L_tpt = avg_entropy(logits[sel_idx])
        self.trace['entropy'].append(L_tpt.item())

        # SAR filter: skip images whose entropy is above 0.4 ln N
        if a.remedy == 'sar' and L_tpt.item() > 0.4 * math.log(T.shape[0]):
            return sel_idx

        need_split = (a.record_mu or a.control == 'random_flip' or a.anneal != 'none'
                      or self.save_trace)
        R, F_pair = None, None
        if self.reg is not None:
            R = self.reg(T)
            if self.reg.gram_based:
                F_pair = self.reg.pair_force(T.detach() @ T.detach().t())

        if R is None:
            g = grad_of(L_tpt, ctx, a.loss_scale, retain=False)
            g_tpt, b = g, torch.zeros_like(g)
        elif need_split:
            g_tpt = grad_of(L_tpt, ctx, a.loss_scale, retain=True)
            g_reg = grad_of(R, ctx, a.loss_scale, retain=False)
            w = self.weight(g_reg, F_pair)
            b = w * g_reg
            g = g_tpt + b
            self.trace['offset_norm'].append(b.norm().item())
            if F_pair is not None:
                self.trace['pair_force'].append(w * F_pair.abs().sum().item())
        else:
            w = a.lam
            g = grad_of(L_tpt + w * R, ctx, a.loss_scale, retain=False)
            g_tpt, b = None, None

        if a.record_mu and g_tpt is not None:
            mu, nu = override_stats(g_tpt, b)
            self.trace['mu'].append(mu)
            self.trace['nu'].append(nu)

        if a.control == 'random_flip':                       # reverse TPT's step on a random set
            g = random_flip_grad(g_tpt, a.flip_mu, self.flip_gen)

        if not torch.isfinite(g).all():
            return sel_idx

        if a.remedy == 'sar':                                # sharpness-aware step (radius 0.05)
            with torch.no_grad():
                e = a.sar_rho * g / g.norm().clamp_min(1e-12)
                ctx.add_(e)
            T2 = model.get_text_features()
            L2 = avg_entropy(model.logits(img_feats, T2)[sel_idx])
            if self.reg is not None:
                L2 = L2 + (w if R is not None else 0.0) * self.reg(T2)
            g = grad_of(L2, ctx, a.loss_scale, retain=False)
            with torch.no_grad():
                ctx.sub_(e)

        ctx.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # diagnostics on the class features at this step
        if F_pair is not None and a.diagnostics:
            S = T.detach() @ T.detach().t()
            od = ~torch.eye(S.shape[0], dtype=torch.bool, device=S.device)
            below = (S <= self.c_star) & od
            self.force_below += F_pair.abs()[below].sum().item()
            self.force_total += F_pair.abs()[od].sum().item()
            self.reached |= below

        if self.ema_ctx is not None:
            with torch.no_grad():
                self.ema_ctx.mul_(a.ema_m).add_((1 - a.ema_m) * ctx.detach())

        if a.remedy == 'sar':                                # reset when the moving entropy collapses
            self.sar_ema = L_tpt.item() if self.sar_ema is None else 0.9 * self.sar_ema + 0.1 * L_tpt.item()
            if self.sar_ema < a.sar_reset:
                self.reset()
                self.sar_ema = None
        return sel_idx

    def adapt_and_predict(self, views):
        a = self.args
        if a.setting == 'episodic' or (a.remedy == 'periodic_reset' and self.t % a.reset_every == 0):
            self.reset()
        img_feats = self.model.encode_image(views)
        if not a.no_tune:
            sel = None
            for _ in range(a.tta_steps):
                sel = self.step(img_feats, sel)
            if self.reg is not None and a.anneal == 'per_pair':
                with torch.no_grad():
                    T = self.model.get_text_features()
                    self.reg.after_step(T @ T.t())
        with torch.no_grad():
            T = self.model.get_text_features(ctx=self.ema_ctx)
            probs = self.model.logits(img_feats[:1], T).softmax(1)[0]
        self.t += 1
        conf, pred = probs.max(0)
        return conf.item(), pred.item()


# ============================================================== run
def run(args):
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset, classnames = get_dataset(args, tpt_transform(args))
    model = get_coop(args.arch, classnames, device, args.n_ctx, args.ctx_init)
    for name, p in model.named_parameters():
        p.requires_grad_(name == "prompt_learner.ctx")
    model.eval()

    with torch.no_grad():
        T0 = model.get_text_features(chunk=args.text_chunk)       # zero-shot class features
    N, D = T0.shape
    print(f"=> {args.test_sets}: {len(dataset)} images, N={N} classes, D={D}")

    ad = Adapter(args, model, N, D, device)
    order = stream_order(args, dataset)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, order), batch_size=1,
                                         shuffle=False, num_workers=args.workers, pin_memory=True)
    confs, preds, labels = [], [], []
    t0 = time.time()
    for i, (images, target) in enumerate(loader):
        views = torch.cat([im.to(device, non_blocking=True) for im in images], 0)
        conf, pred = ad.adapt_and_predict(views)
        confs.append(conf); preds.append(pred); labels.append(int(target))
        if (i + 1) % args.print_freq == 0:
            c = np.asarray(preds) == np.asarray(labels)
            print(f"[{i + 1}/{len(order)}] acc {100 * c.mean():.2f}  "
                  f"ece {100 * metrics.ece(confs, c, args.ece_bins):.2f}  ({time.time() - t0:.0f}s)")

    confs, preds, labels = map(np.asarray, (confs, preds, labels))
    correct = preds == labels
    with torch.no_grad():
        T_end = model.get_text_features(ctx=ad.ema_ctx, chunk=args.text_chunk)

    res = {'config': vars(args), 'N': N, 'D': D, 'n_images': int(len(order)),
           'level': ad.reg.c if ad.reg is not None else None, 'c_star': ad.c_star,
           'acc': 100 * correct.mean(),
           'ece': 100 * metrics.ece(confs, correct, args.ece_bins),
           'ece20': 100 * metrics.ece(confs, correct, 20),
           'gap': 100 * metrics.gap(confs, correct),
           'noise_floor': 100 * metrics.noise_floor(confs, args.ece_bins),
           'debiased_ece': 100 * metrics.debiased_ece(confs, correct, args.ece_bins)}
    sm = metrics.smooth_ece(confs, correct)
    res['smooth_ece'] = None if sm is None else 100 * sm
    if args.setting == 'online':
        if N <= 5000:   # N x N structure measures; skipped for the large ImageNet-21k subsets
            res['spearman_rs'] = metrics.spearman_structure(T0, T_end)
            res['jaccard'] = metrics.jaccard_at_k(T0, T_end, k=3 if N <= 10 else 10)
        res['absorbed_classes'] = metrics.absorbed_classes(preds, labels)
    if ad.reg is not None and ad.reg.half is not None:
        half = ad.reg.half.cpu().numpy()
        res['ece_changed_half'] = 100 * metrics.class_conditional_ece(confs, correct, labels, np.where(half)[0])
        res['ece_other_half'] = 100 * metrics.class_conditional_ece(confs, correct, labels, np.where(~half)[0])
    if args.diagnostics and ad.force_total > 0:
        res['force_below_cstar_pct'] = 100 * ad.force_below / ad.force_total
        od = ~torch.eye(N, dtype=torch.bool, device=device)
        res['reached_pct'] = 100 * ad.reached[od].float().mean().item()
    if ad.trace['mu']:
        res['override_mass_mu'] = float(np.mean(ad.trace['mu']))
        res['override_fraction_nu'] = float(np.mean(ad.trace['nu']))
    if args.eval_end:                                        # ECE_end: frozen final prompt, whole test set
        ds_plain, _ = get_dataset(args, plain_transform(args))
        dl = torch.utils.data.DataLoader(ds_plain, batch_size=256, num_workers=args.workers)
        c_end, k_end = [], []
        with torch.no_grad():
            for x, y in dl:
                p = model.logits(model.encode_image(x.to(device)), T_end).softmax(1)
                cf, pr = p.max(1)
                c_end.append(cf.cpu()); k_end.append((pr.cpu() == y))
        res['ece_end'] = 100 * metrics.ece(torch.cat(c_end), torch.cat(k_end), args.ece_bins)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = args.tag or (f"{args.test_sets}_{args.arch.replace('/', '')}_{args.reg}_{args.setting}"
                       f"_lam{args.lam}_seed{args.seed}_order{args.stream_seed}")
    base = os.path.join(args.save_dir, tag)
    with open(base + '.json', 'w') as fp:
        json.dump(res, fp, indent=1, default=float)
    np.savez_compressed(base + '.npz', conf=confs, pred=preds, label=labels, order=np.asarray(order))
    if ad.save_trace:
        with open(args.trace_out or base + '_trace.json', 'w') as fp:
            json.dump(ad.trace, fp)
    if args.save_features:
        torch.save({'T0': T0.cpu(), 'T_end': T_end.cpu()}, base + '_features.pt')
    print(json.dumps({k: v for k, v in res.items() if k != 'config'}, indent=1, default=float))
    return res


def get_parser():
    p = argparse.ArgumentParser(description='AGI-TPT')
    p.add_argument('data', metavar='DIR', help='dataset root')
    p.add_argument('--test_sets', type=str, default='I', help='one dataset id (I/A/V/R/K, DTD, ..., cifar100, in21k)')
    p.add_argument('--dataset_mode', type=str, default='test')
    p.add_argument('--classnames_file', type=str, default=None, help='in21k: wnid<TAB>name per line')
    p.add_argument('-a', '--arch', default='ViT-B/16')
    p.add_argument('--resolution', default=224, type=int)
    p.add_argument('-j', '--workers', default=4, type=int)
    p.add_argument('-b', '--batch-size', default=64, type=int, help='number of views (1 + augmented)')
    p.add_argument('--lr', default=5e-3, type=float)
    p.add_argument('--wd', default=None, type=float, help='AdamW weight decay (default 0.01 episodic, 0 online)')
    p.add_argument('--loss_scale', default=1000.0, type=float, help='static loss scale for the fp16 encoder')
    p.add_argument('-p', '--print-freq', default=1000, type=int)
    p.add_argument('--gpu', default=0, type=int)
    p.add_argument('--selection_p', default=0.1, type=float)
    p.add_argument('--tta_steps', default=1, type=int)
    p.add_argument('--n_ctx', default=4, type=int)
    p.add_argument('--ctx_init', default='a_photo_of_a', type=str)
    p.add_argument('--I_augmix', action='store_true')
    p.add_argument('--seed', type=int, default=0, help='augmentation seed (episodic: 0-4)')
    p.add_argument('--text_chunk', type=int, default=None, help='encode classes in blocks without grad')
    # setting
    p.add_argument('--setting', choices=['episodic', 'online'], default='episodic')
    p.add_argument('--stream_seed', type=int, default=0, help='stream order (online: 0-9)')
    p.add_argument('--passes', type=int, default=1, help='passes over the stream (Supp. S6.7)')
    p.add_argument('--label_shift', type=float, default=None, help='Dirichlet concentration (UNCERTAIN protocol)')
    p.add_argument('--max_images', type=int, default=None)
    p.add_argument('--no_tune', action='store_true', help='zero-shot CLIP')
    # method
    p.add_argument('--reg', default='agi', choices=['none', 'agi', 'agi_fixed', 'agi_l0', 'agi_lsimplex',
                                                     'agi_nolevel', 'agi_angle', 'agi_nn', 'agi_perclass',
                                                     'ctpt', 'otpt', 'dtpt', 'atpt', 'etf_nc', 'split'])
    p.add_argument('--lam', type=float, default=4.0, help='weight lambda (AGI-TPT: 4, selected on CIFAR-100)')
    p.add_argument('--level_mode', default='attainable',
                   choices=['attainable', 'zero', 'simplex', 'none', 'other_regime', 'welch', 'levenshtein', 'value'])
    p.add_argument('--level_offset', type=float, default=0.0, help='xi: level = c* + xi')
    p.add_argument('--level_value', type=float, default=None)
    p.add_argument('--split_kind', choices=['two_sided', 'level0'], default=None)
    p.add_argument('--split_seed', type=int, default=0)
    # controls
    p.add_argument('--control', choices=['none', 'random_flip'], default='none')
    p.add_argument('--flip_mu', type=float, default=0.58, help="override mass to match (AGI-TPT's on ImageNet: 0.58)")
    p.add_argument('--anneal', choices=['none', 'offset_norm', 'push_mass', 'per_pair'], default='none')
    p.add_argument('--trace_in', type=str, default=None, help="AGI-TPT trace to match (annealed controls)")
    p.add_argument('--trace_out', type=str, default=None)
    p.add_argument('--record_mu', action='store_true', help='record override mass mu and fraction nu')
    # drift remedies
    p.add_argument('--remedy', choices=['none', 'periodic_reset', 'ema', 'sar'], default='none')
    p.add_argument('--reset_every', type=int, default=1000)
    p.add_argument('--ema_m', type=float, default=0.999)
    p.add_argument('--sar_rho', type=float, default=0.05)
    p.add_argument('--sar_reset', type=float, default=0.2)
    # evaluation
    p.add_argument('--ece_bins', type=int, default=15)
    p.add_argument('--eval_end', action='store_true', help='ECE_end of the final prompt')
    p.add_argument('--diagnostics', action='store_true', help='force below c*, reached pairs')
    p.add_argument('--save_features', action='store_true')
    p.add_argument('--save_dir', default='results')
    p.add_argument('--tag', default=None)
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.reg == 'split':
        assert args.split_kind is not None, "--reg split needs --split_kind"
    if args.anneal in ('offset_norm', 'push_mass'):
        assert args.trace_in, "annealed controls need --trace_in from an online AGI-TPT run"
    run(args)
