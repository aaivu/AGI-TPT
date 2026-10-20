#!/bin/bash
# Table 9/10: drift remedies along the stream (online, 10 orders).
#   TPT + periodic reset (every 1000 images), TPT + SAR, A-TPT + {periodic reset, EMA teacher, SAR}
# bash scripts/run_remedies.sh {dataset}
source "$(dirname "$0")/common.sh"
W_A=${W_A:-160}    # A-TPT online weight: x2 its published 80 (appendix G.5)
for order in 0 1 2 3 4 5 6 7 8 9; do
  for r in periodic_reset sar; do
    run $1 --reg none --setting online --stream_seed ${order} --remedy ${r} --eval_end \
        --tag "$1_tpt_${r}_order${order}"
  done
  for r in periodic_reset ema sar; do
    run $1 --reg atpt --lam ${W_A} --setting online --stream_seed ${order} --remedy ${r} --eval_end \
        --tag "$1_atpt_${r}_order${order}"
  done
done
