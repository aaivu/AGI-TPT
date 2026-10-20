#!/bin/bash
# TPT (no regularizer), episodic, 5 seeds. ImageNet and its shifts (I/A/V/R/K).
# bash scripts/test_tpt_ds.sh {I|A|V|R|K}
source "$(dirname "$0")/common.sh"
for seed in 0 1 2 3 4; do run $1 --reg none --setting episodic --seed ${seed}; done
