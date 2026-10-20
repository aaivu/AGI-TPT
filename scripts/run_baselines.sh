#!/bin/bash
# Spread-regularizer baselines (re-implemented written objectives, appendix D).
#   episodic: published weights  C-TPT 50, O-TPT 18, D-TPT 1e5, A-TPT 80
#             (ImageNet shifts: C-TPT 20, O-TPT 2, A-TPT 10)
#   online:   weights selected on held-out CIFAR-100 (appendix G.5):
#             x4 for C-TPT, x2 for O-TPT, D-TPT, A-TPT; plus the neural-collapse ETF target.
# Official-code runs (cells marked with a double dagger) should use each baseline's own repository.
# bash scripts/run_baselines.sh {dataset} {episodic|online}
source "$(dirname "$0")/common.sh"
set=$1; setting=${2:-episodic}
if [[ " ${SHIFTS} " == *" ${set} "* ]]; then W_C=20; W_O=2; W_A=10; else W_C=50; W_O=18; W_A=80; fi
W_D=100000
if [ "${setting}" == "online" ]; then
  W_C=$((W_C * 4)); W_O=$((W_O * 2)); W_A=$((W_A * 2)); W_D=200000
  W_ETF=${W_ETF:-4}    # select on CIFAR-100 with scripts/run_select_lambda.sh
  for order in 0 1 2 3 4 5 6 7 8 9; do
    for m in "ctpt ${W_C}" "otpt ${W_O}" "dtpt ${W_D}" "atpt ${W_A}" "etf_nc ${W_ETF}" "none 0"; do
      set -- ${m}
      run ${set} --reg $1 --lam $2 --setting online --stream_seed ${order} --eval_end --save_features
    done
  done
else
  for seed in 0 1 2 3 4; do
    for m in "ctpt ${W_C}" "otpt ${W_O}" "dtpt ${W_D}" "atpt ${W_A}"; do
      set -- ${m}
      run ${set} --reg $1 --lam $2 --setting episodic --seed ${seed} --record_mu
    done
  done
fi
