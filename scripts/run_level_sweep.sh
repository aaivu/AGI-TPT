#!/bin/bash
# Section 5.4 / appendix G.7, table 12, fig. 4: online AGI-TPT with level c* + xi,
# xi in {-0.05, -0.02, 0, +0.02, +0.05}, and with the other regime's level; 10 orders.
# 18 datasets: the 11 benchmarks, the four ImageNet shifts, fer2013, cifar10, stl10.
# Optional LAM=2 or LAM=8 for the weight check of table S17.
# bash scripts/run_level_sweep.sh {dataset}
source "$(dirname "$0")/common.sh"
LAM=${LAM:-4}
for order in 0 1 2 3 4 5 6 7 8 9; do
  for xi in -0.05 -0.02 0 0.02 0.05; do
    run $1 --reg agi --lam ${LAM} --setting online --stream_seed ${order} --level_offset ${xi} \
        --diagnostics --tag "$1_${ARCH//\//}_level_xi${xi}_lam${LAM}_order${order}"
  done
  run $1 --reg agi --lam ${LAM} --setting online --stream_seed ${order} --level_mode other_regime \
      --tag "$1_${ARCH//\//}_level_other_lam${LAM}_order${order}"
done
