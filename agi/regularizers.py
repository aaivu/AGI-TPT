"""AGI-TPT's loss, its ablations (table 11) and the spread regularizers it is
compared with (appendix D, written objectives).

All regularizers take T, the N x D matrix of unit class features (fp32), and
return a scalar. Pairwise ones are written on the Gram matrix S = T T^T so the
pair forces dR/dS_ij can be read off for the diagnostics of appendix F.5.

AGI-TPT (eq. 1):   L_AGI(T) = 1/N^2 * sum_{i != j} [ReLU(s_ij - c*)]^2
There is no margin and no two-sided term: pairs at or below the level receive
no force (proposition 6).

Baselines here are re-implementations of each paper's written objective, used
for the controls and ablations. Cells marked "double dagger" in the paper were run with each
baseline's official code; use those repositories to reproduce them. O-TPT's
Householder step is not implemented (table 7 reports "without Householder").
"""
import math
import torch
import torch.nn.functional as F

from .level import get_level

GRAM_BASED = {'agi', 'agi_fixed', 'agi_l0', 'agi_lsimplex', 'agi_nolevel', 'agi_angle',
              'agi_nn', 'otpt', 'atpt', 'split'}


def offdiag(N, device):
    return ~torch.eye(N, dtype=torch.bool, device=device)


# ---------------------------------------------------------------- pair losses on S
def relu_residual(S, c, mask=None):
    """1/N^2 sum_{i!=j} [ReLU(s_ij - c)]^2 (AGI-TPT, eq. 1)."""
    N = S.shape[0]
    m = offdiag(N, S.device) if mask is None else mask
    r = F.relu(S - c) * m
    return (r ** 2).sum() / N ** 2


def two_sided_residual(S, c, mask=None):
    """1/N^2 sum_{i!=j} (s_ij - c)^2 (AGI-Fixed; with c = 0 it is O-TPT's Frobenius penalty)."""
    N = S.shape[0]
    m = offdiag(N, S.device) if mask is None else mask
    r = (S - c) * m
    return (r ** 2).sum() / N ** 2


def angle_residual(S, c):
    """AGI-Angle: 1/N^2 sum_{i!=j} [ReLU(theta* - theta_ij)]^2, theta* = arccos(c)."""
    N = S.shape[0]
    theta = torch.arccos(S.clamp(-1 + 1e-6, 1 - 1e-6))
    r = F.relu(math.acos(max(-1.0, min(1.0, c))) - theta) * offdiag(N, S.device)
    return (r ** 2).sum() / N ** 2


def nearest_pair_residual(S, c):
    """AGI-NN: force on each class's nearest pair only, 1/N sum_k [ReLU(max_j s_kj - c)]^2."""
    N = S.shape[0]
    s_max = S.masked_fill(~offdiag(N, S.device), -2.0).max(dim=1).values
    return (F.relu(s_max - c) ** 2).mean()


# ---------------------------------------------------------------- per-class / centroid losses on T
def per_class_residual(T, c):
    """AGI-PerClass: a per-class term with no pair term.
    UNCERTAIN: the paper's table 11 only says "per-class term, no pair term"; here
    each class is pushed from the centroid of the others, [ReLU(<t_k, mean_{j!=k} t_j> - c)]^2."""
    N = T.shape[0]
    others = (T.sum(0, keepdim=True) - T) / (N - 1)
    return (F.relu((T * others).sum(1) - c) ** 2).mean()


def ctpt_loss(T):
    """C-TPT: maximize ATFD = 1/N sum_i ||t_bar - t_i||_2 (Yoon et al. 2024); we minimize -ATFD."""
    return -(T - T.mean(0, keepdim=True)).norm(dim=1).mean()


def otpt_loss(S):
    """O-TPT, Frobenius reading: ||S - I||_F^2 = sum_{i!=j} s_ij^2 (no Householder step)."""
    N = S.shape[0]
    return ((S - torch.eye(N, device=S.device)) ** 2).sum()


def dtpt_loss(T):
    """D-TPT: 1/N sum_k KL(softmax(t_k) || Uniform over D coordinates)."""
    logp = F.log_softmax(T, dim=1)
    return (logp.exp() * (logp + math.log(T.shape[1]))).sum(1).mean()


def atpt_loss(S):
    """A-TPT: minimize -AD, AD = 1/N sum_i min_{j!=i} arccos s_ij."""
    N = S.shape[0]
    theta = torch.arccos(S.clamp(-1 + 1e-6, 1 - 1e-6))
    theta = theta.masked_fill(~offdiag(N, S.device), 10.0)
    return -theta.min(dim=1).values.mean()


def etf_nc_loss(T):
    """Neural-collapse ETF target (appendix F.5): 1/N^2 ||S_bar - N/(N-1)(I - 11^T/N)||_F^2,
    S_bar the Gram matrix of the centred, renormalized features."""
    N = T.shape[0]
    Tb = T - T.mean(0, keepdim=True)
    Tb = Tb / Tb.norm(dim=1, keepdim=True)
    target = N / (N - 1) * (torch.eye(N, device=T.device) - 1.0 / N)
    return ((Tb @ Tb.t() - target) ** 2).sum() / N ** 2


# ---------------------------------------------------------------- wrapper with state
class Regularizer:
    """Callable R(T) with the state needed for the online variants.

    name: agi | agi_fixed | agi_l0 | agi_lsimplex | agi_nolevel | agi_angle | agi_nn |
          agi_perclass | ctpt | otpt | dtpt | atpt | etf_nc | split
    split: split test of section 5.3 / appendix F.5. Pairs with both classes in
           the first half use `split_kind` ('two_sided' at c*, R2; 'level0', R1);
           every other pair keeps AGI-TPT's loss.
    per_pair_decay: annealed control, each pair's weight 1/(1 + n_ij/1000),
           n_ij = number of steps pair (i, j) has been pushed.
    """

    def __init__(self, name, N, D, level_mode='attainable', level_offset=0.0, level_value=None,
                 split_kind=None, split_seed=0, per_pair_decay=False, device='cuda'):
        self.name, self.N, self.D = name, N, D
        defaults = {'agi_l0': 'zero', 'agi_lsimplex': 'simplex', 'agi_nolevel': 'none'}
        mode = defaults.get(name, level_mode)
        self.c = get_level(N, D, mode, level_offset, level_value) if N > 1 else -1.0
        self.c_star = get_level(N, D, 'attainable') if N > 1 else -1.0
        self.per_pair_decay = per_pair_decay
        self.n_pushed = torch.zeros(N, N, device=device) if per_pair_decay else None
        self.half = None
        if name == 'split':
            g = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(N, generator=g)
            self.half = torch.zeros(N, dtype=torch.bool)
            self.half[perm[: N // 2]] = True
            self.half = self.half.to(device)
            self.split_kind = split_kind
        print(f"[regularizer] {name}: N={N}, D={D}, level c={self.c:.4f} (c*={self.c_star:.4f})")

    @property
    def gram_based(self):
        return self.name in GRAM_BASED

    def on_gram(self, S):
        n, c = self.name, self.c
        if n in ('agi', 'agi_l0', 'agi_lsimplex', 'agi_nolevel'):
            if self.per_pair_decay:
                N = S.shape[0]
                w = 1.0 / (1.0 + self.n_pushed / 1000.0)
                r = F.relu(S - c) * offdiag(N, S.device)
                return (w * r ** 2).sum() / N ** 2
            return relu_residual(S, c)
        if n == 'agi_fixed':
            return two_sided_residual(S, c)
        if n == 'agi_angle':
            return angle_residual(S, c)
        if n == 'agi_nn':
            return nearest_pair_residual(S, c)
        if n == 'otpt':
            return otpt_loss(S)
        if n == 'atpt':
            return atpt_loss(S)
        if n == 'split':
            N = S.shape[0]
            od = offdiag(N, S.device)
            inside = self.half[:, None] & self.half[None, :] & od
            outside = od & ~inside
            if self.split_kind == 'two_sided':
                alt = two_sided_residual(S, self.c_star, inside)
            elif self.split_kind == 'level0':
                alt = relu_residual(S, 0.0, inside)
            else:
                raise ValueError(self.split_kind)
            return relu_residual(S, self.c_star, outside) + alt
        raise ValueError(n)

    def __call__(self, T):
        if self.gram_based:
            return self.on_gram(T @ T.t())
        if self.name == 'agi_perclass':
            return per_class_residual(T, self.c)
        if self.name == 'ctpt':
            return ctpt_loss(T)
        if self.name == 'dtpt':
            return dtpt_loss(T)
        if self.name == 'etf_nc':
            return etf_nc_loss(T)
        raise ValueError(self.name)

    @torch.no_grad()
    def after_step(self, S):
        if self.per_pair_decay:
            self.n_pushed += ((S > self.c) & offdiag(S.shape[0], S.device)).float()

    def pair_force(self, S):
        """dR/dS (N x N) for Gram-based losses, else None (diagnostics, annealing)."""
        if not self.gram_based:
            return None
        with torch.enable_grad():
            S_leaf = S.detach().clone().requires_grad_(True)
            g, = torch.autograd.grad(self.on_gram(S_leaf), S_leaf)
        return g
