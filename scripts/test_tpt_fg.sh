#!/bin/bash
# TPT (no regularizer), episodic, 5 seeds. Fine-grained datasets.
# bash scripts/test_tpt_fg.sh {dataset}
source "$(dirname "$0")/common.sh"
for seed in 0 1 2 3 4; do run $1 --reg none --setting episodic --seed ${seed}; done
