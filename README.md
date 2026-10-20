# AGI-TPT: Adaptive Geometry-Informed Test-Time Prompt Tuning

Code for *AGI-TPT: Adaptive Geometry-Informed Test-Time Prompt Tuning for Calibrated
Zero-Shot Vision-Language Models* (under review, ICLR 2027).

Test-time prompt tuning (TPT) adapts CLIP to each test image by lowering the entropy of
its prediction. This raises accuracy but makes the model over-confident, and when the
prompt is carried along a stream of test images it collapses. AGI-TPT adds one
regularizer on the Gram matrix `S = T Tᵀ` of the unit class text features:

```
L_AGI(T) = 1/N² · ‖ReLU(S − G_c*)‖²_F = 1/N² · Σ_{i≠j} [ReLU(s_ij − c*)]²
```

* **Where to stop (R1):** the *attainable level* `c*(N, D)` is the smallest largest
  pairwise cosine that `N` unit vectors in `R^D` can have:
  `−1/(N−1)` for `N ≤ D+1`, `0` for `D+2 ≤ N ≤ 2D`, and a certified upper bound `U` beyond `2D`.
  It is computed from `(N, D)`, never tuned (`agi/level.py`).
* **Which pairs to move (R2):** the ReLU is applied *before* squaring, so only pairs whose
  cosine is above the level are pushed. Pairs at or below it receive no force.
  There is no margin term and no two-sided residual.

The total loss is `L_TPT + λ·L_AGI` with **λ = 4** for every dataset, backbone and setting
(selected once on held-out CIFAR-100). The only extra cost is one `N×N` Gram product per step.

## Installation
```bash
git clone https://github.com/aaivu/AGI-TPT.git
cd AGI-TPT
conda env create -f environment.yml
conda activate ctpt
pip install relplot   # optional: smooth ECE
```

## Datasets
Prepare the datasets as in [TPT](https://github.com/azshue/TPT) and set `DATA_ROOT` in
`scripts/common.sh` (or export it).

| Group | Ids (`--test_sets`) |
|---|---|
| Benchmarks (11) | `I` (ImageNet), `DTD`, `Flower102`, `Food101`, `SUN397`, `Aircraft`, `Pets`, `Caltech101`, `UCF101`, `eurosat`, `Cars` |
| ImageNet shifts | `A`, `V`, `R`, `K` (Sketch) |
| Level tests / λ selection | `fer2013`, `cifar10`, `stl10`, `cifar100` (torchvision formats under `DATA_ROOT`) |
| Beyond 2D | `in21k` + `--classnames_file` (lines `wnid<TAB>name`; images in `DATA_ROOT/imagenet21k/<wnid>/`) |

## Repository layout
```
agi_tpt_classification.py   main script: one run = one dataset × method × setting × seed/order
agi/
  level.py                  attainable level c*(N,D), certified U, Welch, ablation levels
  regularizers.py           AGI-TPT loss, table-11 ablations, C-/O-/D-/A-TPT, ETF target, split test
  strength.py               override set / mass μ (prop. 1, cor. 2), random flip control
  metrics.py                ECE, gap, noise floor, debiased/smooth ECE, class-conditional ECE,
                            paired bootstrap, Spearman r_s, Jaccard@k, absorbed classes
experiments/
  equal_strength.py         match every regularizer's override mass to AGI-TPT's (table 7)
  entropy_pulls.py          entropy pulls competitors together (prop. 3, fig. 3a)
analysis/aggregate.py       mean±s.d. tables, paired bootstrap ("resolved"), cumulative ECE curves
scripts/                    one script per experiment of the paper (below)
clip/, data/, utils/        CLIP, datasets and helpers (from TPT / C-TPT)
```

## Settings
* **Episodic** (`--setting episodic`): one AdamW step per image (lr 5e-3, weight decay 0.01),
  then reset to "a photo of a"; 64 views, the 10% most confident in the entropy; 5 seeds.
* **Online** (`--setting online`): one step per image without reset, weight decay 0;
  10 stream orders (`--stream_seed 0..9`) shared by all methods.
* ECE uses 15 equal-width bins (`--ece_bins`); 20-bin ECE is also saved (`ece20`).

## Running the experiments
| Paper | Script |
|---|---|
| Zero-shot CLIP, TPT | `scripts/test_baseline.sh`, `scripts/test_tpt_fg.sh`, `scripts/test_tpt_ds.sh` |
| Table 1 / 6, episodic AGI-TPT | `scripts/run_agi_episodic.sh {dataset}` (`ARCH=RN50` or `ARCH=ViT-L/14` for table 6) |
| Table 1 / 8, online AGI-TPT, fig. 1b, fig. 3b | `scripts/run_agi_online.sh {dataset}` |
| Baselines (episodic / online) | `scripts/run_baselines.sh {dataset} {episodic\|online}`; C-TPT only: `scripts/test_tpt_ctpt_{fg,ds}.sh` |
| Table 7, equal strength + random flip set | `scripts/run_equal_strength.sh {dataset}` |
| Table 9 / 10, drift remedies (periodic reset, EMA teacher, SAR) | `scripts/run_remedies.sh {dataset}` |
| Table 11, ablations + annealed controls | `scripts/run_ablations.sh {dataset}` (after `run_agi_online.sh`) |
| Table 12, fig. 4, level `c* + ξ` on 18 datasets | `scripts/run_level_sweep.sh {dataset}` |
| Table 13, ImageNet-21k subsets (beyond 2D) | `scripts/run_in21k.sh {subset_file}` |
| Fig. 3c, causal split tests | `scripts/run_split_tests.sh` |
| Fig. 3a, label shift, three passes | `scripts/run_mechanism.sh` |
| Table 5, λ selection on CIFAR-100 | `scripts/run_select_lambda.sh` |

Single run, for example online AGI-TPT on DTD:
```bash
python agi_tpt_classification.py $DATA_ROOT --test_sets DTD -a ViT-B/16 \
    --reg agi --lam 4 --setting online --stream_seed 0 --eval_end --diagnostics
```

Each run writes `results/<tag>.json` with these metrics: accuracy, ECE, ECE (20 bins), gap,
noise floor, debiased ECE, smooth ECE, ECE_end, Spearman r_s, Jaccard@k, absorbed classes,
force below c*, reached pairs, μ/ν and split-half ECE. It also writes `results/<tag>.npz`
with the per-image confidence, prediction, label and stream order. Then:
```bash
python analysis/aggregate.py summary results/ --out results/summary.csv
python analysis/aggregate.py compare results/ --a DTD_ViT-B16_agi_online --b DTD_ViT-B16_atpt_online
python analysis/aggregate.py curve results/DTD_ViT-B16_agi_online_lam4.0_seed0_order0.npz
```

## Method options at a glance
| `--reg` | Meaning |
|---|---|
| `agi` | AGI-TPT: `[ReLU(s − c*)]²`, level c*(N,D) |
| `agi_fixed` | two-sided residual `(s − c*)²` |
| `agi_l0` | level 0 (O-TPT's), with ReLU |
| `agi_lsimplex` | level `−1/(N−1)`, ignores D |
| `agi_nolevel` | no level (`c = −1`), never stops; with `--anneal offset_norm\|push_mass\|per_pair` |
| `agi_angle` | `[ReLU(θ* − θ_ij)]²` in angle |
| `agi_nn` | force on each class's nearest pair only |
| `agi_perclass` | per-class term, no pair term |
| `ctpt`, `otpt`, `dtpt`, `atpt` | written objectives of C-TPT, O-TPT (Frobenius, no Householder), D-TPT, A-TPT |
| `etf_nc` | neural-collapse simplex-ETF target on centred features |
| `split` | split test (`--split_kind two_sided\|level0`) |

Levels: `--level_mode attainable|zero|simplex|none|other_regime|welch|levenshtein|value`,
`--level_offset ξ`.

## Notes on fidelity
* Published baseline numbers in the paper are copied from each paper. The `ctpt/otpt/dtpt/atpt`
  options re-implement the written objectives for controls and ablations. Use the official
  repositories to reproduce the official-code cells.
* `agi/level.py` builds U from Prop. B.3, Prop. B.4 and real mutually unbiased bases. These
  reproduce every U in table 13. The supplement's other certificates are not included, and
  Levenshtein's bound is available only for the `(D, N)` printed in table 13.
* `agi_perclass`, the debiased-ECE variant and the label-shift ordering follow our reading of
  the paper and are marked `UNCERTAIN` in the code.
