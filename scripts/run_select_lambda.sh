#!/bin/bash
# Appendix F.4, table 5: select lambda on held-out CIFAR-100 (not a benchmark),
# by the ECE of one TPT step on its test split and of an online pass over it.
# Baselines: the same selection over {1/4, 1/2, 1, 2, 4} x published weight, e.g.
#   METHOD=atpt BASE=80 bash scripts/run_select_lambda.sh
source "$(dirname "$0")/common.sh"
METHOD=${METHOD:-agi}
if [ "${METHOD}" == "agi" ]; then GRID="0.25 0.5 1 2 4 8 16 32"
else GRID=$(python -c "print(' '.join(str(m*${BASE}) for m in (0.25,0.5,1,2,4)))"); fi
for lam in ${GRID}; do
  run cifar100 --reg ${METHOD} --lam ${lam} --setting episodic --seed 0 --tag "cifar100_${METHOD}_episodic_lam${lam}"
  run cifar100 --reg ${METHOD} --lam ${lam} --setting online --stream_seed 0 --tag "cifar100_${METHOD}_online_lam${lam}"
done
