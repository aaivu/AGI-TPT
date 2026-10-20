"""One-step strength: override set, override mass and the random flip control.

Section 2 (proposition 1, corollary 2) and appendix F.5.
At Adam's first step every coordinate moves by -eta * sign(g_j + b_j); the
regularizer changes the prediction only on the override set
O_x = {j : sign(g_x,j + b_R,j) != sign(g_x,j)}.
"""
import torch


def override_stats(g, b):
    """Override mass mu = ||g_O||_1 / ||g||_1 and override fraction nu = |O| / d_pi."""
    g, b = g.flatten().float(), b.flatten().float()
    O = (torch.sign(g + b) != torch.sign(g)) & (g != 0)
    mu = (g.abs() * O).sum() / g.abs().sum().clamp_min(1e-30)
    return mu.item(), O.float().mean().item()


def random_flip_grad(g, mu, generator=None):
    """Random flip set: in a random order of the coordinates, reverse TPT's step on
    the first coordinates whose share of ||g||_1 reaches `mu`. Returns the gradient
    to hand to AdamW (its first step is the sign step of this vector)."""
    flat = g.flatten()
    perm = torch.randperm(flat.numel(), generator=generator, device='cpu').to(flat.device)
    share = torch.cumsum(flat.abs()[perm], 0) / flat.abs().sum().clamp_min(1e-30)
    n_flip = int(torch.searchsorted(share, torch.tensor([mu], device=share.device)).item()) + 1
    sign = torch.ones_like(flat)
    sign[perm[:n_flip]] = -1.0
    return (flat * sign).view_as(g)
