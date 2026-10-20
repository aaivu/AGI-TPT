#!/bin/bash
# C-TPT re-implementation on ImageNet shifts at its published weight 20.
# bash scripts/test_tpt_ctpt_ds.sh {I|A|V|R|K}
source "$(dirname "$0")/common.sh"
for seed in 0 1 2 3 4; do run $1 --reg ctpt --lam 20 --setting episodic --seed ${seed}; done
