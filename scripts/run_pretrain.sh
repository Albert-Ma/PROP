#!/usr/bin/env bash

DATA_DIR=data/wiki_info/train
Bert_MODEL_DIR=/data/maxinyu/bert-base-uncased-py
OUTPUT=pretrained_models/

python -m run_pretraining \
        --pregenerated_data $DATA_DIR \
        --bert_model $Bert_MODEL_DIR \
        --do_lower_case \
        --output_dir $OUTPUT \
        --train_batch_size 10\
        --save_checkpoints_steps 1000 \
        --gradient_accumulation_steps 1 \
        --epochs 2 \
        --negtive_num 1 \
        --learning_rate 2e-5 \
        --temp_dir ./