#!/usr/bin/env bash

DATA_DIR=data/wiki_info/bprop
Bert_MODEL_DIR=/data/maxinyu/bert-base-uncased-py


python -m prop.multiprocessing_generate_pairwise_instances \
    --train_corpus ${DATA_DIR} \
    --output_dir ${DATA_DIR}/train \
    --bert_model ${Bert_MODEL_DIR} \
    --do_lower_case \
    --rop_num_per_doc 5 \
    --epochs_to_generate 2 \
    --mlm \
    --max_seq_len 512 \
    --temp_dir ./  \
    --num_workers 2 