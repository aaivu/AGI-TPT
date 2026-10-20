#!/bin/bash
# Table 11: one component of AGI-TPT changed at a time, at AGI-TPT's weight (lambda = 4).
# One step (episodic, seed 0-4) and online (orders 0-9). Run run_agi_online.sh first:
# the annealed controls match the trace of the online AGI-TPT run with the same order.
# bash scripts/run_ablations.sh {dataset}
source "$(dirname "$0")/common.sh"
set=$1
VARIANTS="agi_fixed agi_l0 agi_lsimplex agi_nolevel agi_angle agi_nn agi_perclass"
for seed in 0 1 2 3 4; do
  for v in ${VARIANTS}; do run ${set} --reg ${v} --lam 4 --setting episodic --seed ${seed}; done
done
for order in 0 1 2 3 4 5 6 7 8 9; do
  for v in ${VARIANTS}; do
    run ${set} --reg ${v} --lam 4 --setting online --stream_seed ${order} --diagnostics --save_features
  done
  TRACE=${SAVE_DIR}/${set}_${ARCH//\//}_agi_online_lam4.0_seed0_order${order}_trace.json
  # AGI-NoLevel annealed: offset norm / summed pair force matched to AGI-TPT, or per-pair decay
  run ${set} --reg agi_nolevel --setting online --stream_seed ${order} --diagnostics --save_features \
      --anneal offset_norm --trace_in ${TRACE} --tag ${set}_nolevel_anneal_offset_order${order}
  run ${set} --reg agi_nolevel --setting online --stream_seed ${order} --diagnostics --save_features \
      --anneal push_mass --trace_in ${TRACE} --tag ${set}_nolevel_anneal_pushmass_order${order}
  run ${set} --reg agi_nolevel --lam 4 --setting online --stream_seed ${order} --diagnostics --save_features \
      --anneal per_pair --tag ${set}_nolevel_anneal_perpair_order${order}
  # push-mass-matched level 0 (appendix F.5)
  run ${set} --reg agi_l0 --setting online --stream_seed ${order} --diagnostics \
      --anneal push_mass --trace_in ${TRACE} --tag ${set}_l0_pushmass_order${order}
done
