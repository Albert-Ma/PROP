#!/usr/bin/env bash

INPUT_FILE=./data/wiki_info
Bert_MODEL_DIR=../bert-base-uncased-py

python -m prop.preprocessing_data \
    --corpus_name wikipedia \
    --data_file ${INPUT_FILE}/wiki_info/wiki_toy.data \
    --bert_model ${Bert_MODEL_DIR} \
    --do_lower_case \
    --output_dir ${INPUT_FILE}/wiki_info/