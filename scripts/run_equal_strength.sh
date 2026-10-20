#!/bin/bash
# Section 5.1, table 7: one step at equal strength.
# 1) match every regularizer's override mass mu to AGI-TPT's on 500 training images (no labels);
# 2) run each at its matched weight (episodic, 5 seeds), plus the random flip set (20 draws)
#    at AGI-TPT's override mass.
# Needs `jq` to read the JSON written in step 1.
# bash scripts/run_equal_strength.sh {dataset}
source "$(dirname "$0")/common.sh"
set=$1; OUT=${SAVE_DIR}/equal_strength_${set}.json
python experiments/equal_strength.py ${DATA_ROOT} --test_sets ${set} --arch ${ARCH} --gpu ${GPU} \
    --methods ctpt otpt dtpt atpt --init_weights 50 18 100000 80 --out ${OUT}
MU=$(jq '.agi.mu' ${OUT})
for m in ctpt otpt dtpt atpt; do
  W=$(jq ".${m}.weight" ${OUT})
  for seed in 0 1 2 3 4; do
    run ${set} --reg ${m} --lam ${W} --setting episodic --seed ${seed} --record_mu \
        --tag "${set}_${m}_equalstrength_seed${seed}"
  done
done
for draw in $(seq 0 19); do
  run ${set} --reg none --control random_flip --flip_mu ${MU} --setting episodic --seed ${draw} \
      --tag "${set}_randomflip_seed${draw}"
done
