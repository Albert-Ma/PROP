#!/usr/bin/env bash

DATA_DIR=data/wiki_info/bprop

class="$1"

if [ $class == 'prop' ]
then
    python -m prop.multiprocessing_generate_word_sets \
        --corpus_info_dir ${DATA_DIR}  \
        --output_dir ${DATA_DIR} \
        --epochs_to_generate 2 \
        --rop_num_per_doc 10 \
        --stem \
        --num_workers 2

elif [ $class == 'bprop' ]
then
    python -m bprop.multiprocessing_generate_word_sets \
        --train_corpus ${DATA_DIR}  \
        --output_dir ${DATA_DIR} \
        --do_lower_case \
        --epochs_to_generate 2 \
        --rop_num_per_doc 10 \
        --method entropy \
        --stem \
        --num_workers 2

else
    echo "Invalid argument: $class" 
fi