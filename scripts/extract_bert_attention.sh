#!/usr/bin/env bash

DATA_DIR=./data/wiki_info/
Bert_MODEL_DIR=/data/maxinyu/bert-base-uncased-py
DATA_FILE=./data/wiki_info/wiki_toy.data

index=0

python -m bprop.extract_bert_attention \
        --corpus_name wikipedia \
        --input_file $DATA_FILE \
        --index ${index} \
        --doc_chunk 10 \
        --output_file ${DATA_DIR}/bprop/cls_vanilla_attention_layer12_index-${index}.hdf5 \
        --bert_model $Bert_MODEL_DIR \
        --do_lower_case \
        --sub2whole first \
        --cls \
        --max_seq_length 100 \
        --output_attentions \
        --batch_size 1