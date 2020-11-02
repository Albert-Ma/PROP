#!/usr/bin/env bash
OUTPUT=/data/maxinyu/PROP/pretrained_models/
Bert_MODEL_DIR=/data/maxinyu/bert-base-uncased-py

python run_pretraining.py \
        --pregenerated_data ../data/test \
        --bert_model $Bert_MODEL_DIR \
        --do_lower_case \
        --output_dir $OUTPUT/test/ \
        --train_batch_size 40\
        --save_checkpoints_steps 10\
        --gradient_accumulation_steps 2\
        --epochs 2\
        --negtive_num 1\
        --learning_rate 5e-5\
        --reduce_memory