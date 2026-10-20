#!/bin/bash
# Shared settings for every AGI-TPT script. Edit DATA_ROOT once here (or export it).
DATA_ROOT=${DATA_ROOT:-'put your dataset path here'}
ARCH=${ARCH:-ViT-B/16}          # ViT-B/16 (D=512), RN50 (D=1024), ViT-L/14 (D=768)
GPU=${GPU:-0}
SAVE_DIR=${SAVE_DIR:-results}
MAIN=./agi_tpt_classification.py

FINE_GRAINED="DTD Flower102 Food101 Cars SUN397 Aircraft Pets Caltech101 UCF101 eurosat"
SHIFTS="A V R K"
ONLINE_SIX="I SUN397 Cars Aircraft DTD eurosat"     # table 9 datasets

# Views: 63 random resized crops (+ AugMix on the fine-grained sets, as in TPT; appendix F.5).
run() {   # run <dataset> <extra args...>
  local set=$1; shift
  python ${MAIN} ${DATA_ROOT} --test_sets ${set} -a ${ARCH} --gpu ${GPU} \
    --ctx_init a_photo_of_a -b 64 --lr 5e-3 --save_dir ${SAVE_DIR} "$@"
}
