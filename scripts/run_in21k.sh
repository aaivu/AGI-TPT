#!/bin/bash
# Appendix G.8, table 13: online TPT on ImageNet-21k class subsets, several levels.
# Needs <DATA_ROOT>/imagenet21k/<wnid>/*.JPEG and a class-subset file per N
# (lines "wnid<TAB>class name"), e.g. subsets/in21k_N5000.txt.
# Levels: -1 (no level), 0, Levenshtein L (only (D, N) of table 13), c* or certified U, Welch.
# bash scripts/run_in21k.sh subsets/in21k_N5000.txt
source "$(dirname "$0")/common.sh"
SUBSET=$1; name=$(basename ${SUBSET} .txt)
for order in 0 1 2 3 4 5 6 7 8 9; do
  for lv in "none" "zero" "levenshtein" "attainable" "welch"; do
    run in21k --classnames_file ${SUBSET} --reg agi --lam 4 --setting online --stream_seed ${order} \
        --level_mode ${lv} --text_chunk 2000 --diagnostics \
        --tag "${name}_${ARCH//\//}_level_${lv}_order${order}" || true   # some levels are undefined for some N
  done
done
