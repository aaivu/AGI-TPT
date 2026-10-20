#!/bin/bash
# Zero-shot CLIP ("a photo of a"), no tuning.
# bash scripts/test_baseline.sh {I|A|V|R|K|DTD|Flower102|Food101|Cars|SUN397|Aircraft|Pets|Caltech101|UCF101|eurosat}
source "$(dirname "$0")/common.sh"
run $1 --reg none --no_tune --tag "$1_${ARCH//\//}_zeroshot"
