#!/bin/bash
# AGI-TPT, episodic (one step per image, then reset), lambda = 4, level c*(N, D); 5 seeds.
# Table 1 (episodic block), table 6 (ARCH=RN50 / ViT-L/14), table 7 (with --record_mu).
# bash scripts/run_agi_episodic.sh {dataset}
source "$(dirname "$0")/common.sh"
for seed in 0 1 2 3 4; do
  run $1 --reg agi --lam 4 --setting episodic --seed ${seed} --record_mu
done
