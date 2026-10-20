#!/bin/bash
# C-TPT re-implementation (written objective, appendix D.2) at its published weight 50.
# Published C-TPT numbers in table 1 are copied from Yoon et al. (2024), not from this script.
# bash scripts/test_tpt_ctpt_fg.sh {dataset}
source "$(dirname "$0")/common.sh"
for seed in 0 1 2 3 4; do run $1 --reg ctpt --lam 50 --setting episodic --seed ${seed}; done
