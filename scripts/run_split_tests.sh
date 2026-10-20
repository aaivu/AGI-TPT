#!/bin/bash
# Section 5.3, fig. 3c: causal split tests inside one online run.
# Classes are split at random into two halves (2 partitions x 10 orders); only pairs with
# both classes in the first half get the changed loss. Reports class-conditional ECE per half.
#   (R2) two-sided residual at c*, on ImageNet      (R1) level 0, on EuroSAT
# bash scripts/run_split_tests.sh
source "$(dirname "$0")/common.sh"
for part in 0 1; do
  for order in 0 1 2 3 4 5 6 7 8 9; do
    run I --reg split --split_kind two_sided --split_seed ${part} --lam 4 --setting online \
        --stream_seed ${order} --diagnostics --tag "I_split_twosided_p${part}_order${order}"
    run eurosat --reg split --split_kind level0 --split_seed ${part} --lam 4 --setting online \
        --stream_seed ${order} --diagnostics --tag "eurosat_split_level0_p${part}_order${order}"
  done
done
