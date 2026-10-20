#!/bin/bash
# Fig. 3a / proposition 3: one entropy step pulls competitive classes together
# (free class features and the 4-token prompt), on ImageNet.
# bash scripts/run_mechanism.sh
source "$(dirname "$0")/common.sh"
python experiments/entropy_pulls.py ${DATA_ROOT} --test_sets I --arch ${ARCH} --gpu ${GPU} \
    --n_images 1000 --out ${SAVE_DIR}/entropy_pulls_I.json
# Fig. 3b (Spearman r_s and Jaccard@10) comes from the online runs: see analysis/aggregate.py summary.
# Sustained adaptation (label shift, three passes; Supp. S6.7):
for order in 0 1 2 3 4 5 6 7 8 9; do
  run I --reg agi --lam 4 --setting online --stream_seed ${order} --label_shift 0.1 --tag "I_agi_labelshift_order${order}"
  run I --reg agi --lam 4 --setting online --stream_seed ${order} --passes 3 --tag "I_agi_3passes_order${order}"
done
