"""The attainable level c*(N, D) and the levels used in the ablations.

Paper: section 4.1, appendix B (propositions 11, B.3, B.4), tables 2, 4 and 13.

  regime I    N <= D+1         c* = -1/(N-1)   (regular simplex, exact)
  regime II   D+2 <= N <= 2D   c* = 0          (exact, Rankin)
  regime III  N > 2D           c* unknown; AGI-TPT uses a certified upper bound U >= c*

The certified U is the smallest largest-cosine among the explicit constructions
of appendix B.4 that this file can build from (N, D):
  * Prop. B.3  orthogonal sum of k antipodal simplex ETFs: 2D+2k vectors, 1/floor(D/k)
  * Prop. B.4  antipodal Paley ETF in a d-dim block (+ cross-polytope elsewhere):
               2D+2d vectors, (2d-1)^(-1/2), 2d-1 a prime power = 1 (mod 4)
  * real mutually unbiased bases (Kerdock), table 2 row IIIc: in a block of
               dimension D' = 4^j <= D, 2 D'(D'/2+1) vectors at cosine D'^(-1/2),
               plus a cross-polytope on the other D-D' coordinates.
This reproduces every U printed in table 13 (checked by hand, e.g. D=512, N=1280:
q=1021 -> 0.0313; D=768, N=2000: q=1493 -> 0.0259; N=21841 -> 1/16 for D=512, 768
and 1/32 for D=1024). The supplement's Steiner/Gold certificates are NOT
implemented here, so for other (N, D) this U can be looser than the paper's.
Levenshtein's lower bound L is not computed; only the values printed in
table 13 are available (PAPER_L).
"""
import math

# Levenshtein lower bounds L printed in table 13 ("<1e-4" stored as 0.0).
PAPER_L = {
    (512, 1026): 0.0, (512, 1280): 0.0010, (512, 2000): 0.0037, (512, 5000): 0.0136, (512, 21841): 0.0333,
    (768, 1538): 0.0, (768, 2000): 0.0008, (768, 5000): 0.0057, (768, 21841): 0.0216,
    (1024, 2050): 0.0, (1024, 5000): 0.0028, (1024, 21841): 0.0146,
}


def _is_prime_power(q):
    if q < 2:
        return False
    p = next(p for p in range(2, int(math.isqrt(q)) + 2) if q % p == 0 or p * p > q)
    if q % p != 0:
        return True          # q is prime
    while q % p == 0:
        q //= p
    return q == 1


def certified_upper_bound(N, D):
    """U(N, D) >= c*(N, D) for N > 2D (see module docstring)."""
    assert N > 2 * D
    best = math.inf
    # Prop. B.3: k antipodal simplex ETFs, 2D + 2k vectors, k <= D/2.
    k = math.ceil((N - 2 * D) / 2)
    if k <= D // 2:
        best = min(best, 1.0 / (D // k))
    # Prop. B.4: one antipodal Paley block of dimension d <= D; take the largest q.
    for q in range(2 * D - 1, 0, -1):
        d = (q + 1) // 2
        if q % 4 == 1 and 2 * D + 2 * d >= N and _is_prime_power(q):
            best = min(best, q ** -0.5)
            break
    # Real MUBs (Kerdock) in a block of dimension 4^j.
    Dp = 1
    while Dp * 4 <= D:
        Dp *= 4
    if 2 * Dp * (Dp // 2 + 1) + 2 * (D - Dp) >= N:
        best = min(best, Dp ** -0.5)
    if best is math.inf:
        raise ValueError(f"no certificate implemented for N={N}, D={D}")
    return best


def attainable_level(N, D):
    """c-bar(N, D): c* for N <= 2D, the certified U beyond (table 2)."""
    if N <= 1:
        return -1.0
    if N <= D + 1:
        return -1.0 / (N - 1)
    if N <= 2 * D:
        return 0.0
    return certified_upper_bound(N, D)


def welch(N, D):
    """Welch value W(N, D) = sqrt((N-D)/(D(N-1))), defined for N > D."""
    return math.sqrt((N - D) / (D * (N - 1)))


def get_level(N, D, mode='attainable', offset=0.0, value=None):
    """Level c used by the loss.

    mode:
      attainable   c*(N,D) (U beyond 2D)                      AGI-TPT
      zero         0 (O-TPT's implicit level)                 AGI-L0
      simplex      -1/(N-1), ignores D                        AGI-Lsimplex
      none         -1: never stops                            AGI-NoLevel
      other_regime 0 if N <= D+1, -1/(N-1) if D+2<=N<=2D       fig. 4b / table 12
      welch        W(N, D)                                    appendix G.8
      levenshtein  L from table 13 (only printed (D, N))      appendix G.8
      value        the explicit `value`
    offset: xi of section 5.4 (level = c* + xi).
    """
    if mode == 'attainable':
        c = attainable_level(N, D)
    elif mode == 'zero':
        c = 0.0
    elif mode == 'simplex':
        c = -1.0 / (N - 1)
    elif mode == 'none':
        c = -1.0
    elif mode == 'other_regime':
        if N <= D + 1:
            c = 0.0
        elif N <= 2 * D:
            c = -1.0 / (N - 1)
        else:
            raise ValueError("other_regime is defined only for N <= 2D")
    elif mode == 'welch':
        c = welch(N, D)
    elif mode == 'levenshtein':
        if (D, N) not in PAPER_L:
            raise ValueError(f"Levenshtein bound not available for D={D}, N={N}")
        c = PAPER_L[(D, N)]
    elif mode == 'value':
        c = float(value)
    else:
        raise ValueError(mode)
    return c + offset
