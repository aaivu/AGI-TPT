#!/bin/bash
# AGI-TPT, online (prompt carried along the stream, weight decay 0), 10 stream orders.
# Table 1 (online block), table 8 (ECE, noise floor, ECE_end, debiased), fig. 1b,
# fig. 3b (Spearman r_s, Jaccard@10), table 11 diagnostics (force below c*, reached).
# The run also saves <tag>_trace.json used by the annealed controls (run_ablations.sh).
# bash scripts/run_agi_online.sh {dataset}
source "$(dirname "$0")/common.sh"
for order in 0 1 2 3 4 5 6 7 8 9; do
  run $1 --reg agi --lam 4 --setting online --stream_seed ${order} --eval_end --diagnostics --save_features
done
