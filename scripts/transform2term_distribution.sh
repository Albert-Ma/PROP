#!/usr/bin/env bash

DATA_DIR=./data/wiki_info/bprop

python -m bprop.transform_vanilla_attention_to_term_distribution \
        --train_corpus ${DATA_DIR} \
        --output_dir ${DATA_DIR} \
        --do_lower_case \
        --aggregate sum \
        --saturation k \
        --stem