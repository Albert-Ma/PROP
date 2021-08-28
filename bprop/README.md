# B-PROP

## Introduction
Despite the exciting performance achieved by PROP, its effectiveness might be bounded by the classical unigram language model adopted in the ROP task construction process. To tackle this problem, we propose a bootstrapped pre-training method (namely B-PROP) based on BERT for ad-hoc retrieval. The key idea is to use the powerful contextual language model BERT to replace the classical unigram language model for the ROP task construction, and re-train BERT itself towards the tailored objective for IR. Specifically, we introduce a novel contrastive method, inspired by the divergence-from-randomness idea, to leverage BERT's self-attention mechanism to sample representative words from the document. The full paper can be found [here](https://arxiv.org/pdf/2104.09791.pdf).


## Contrastive Sampling for ROP with BERT

In the following, we first compute *the document term distribution* based on BERT’s vanilla [CLS]-Token attention.
We then take an expectation over all the term distributions in the collection to approximate the term distribution produced by a random process. 
Finally, we compute the cross-entropy (i.e., the divergence) between the two distributions, i.e., *the document term distribution* and *the random term distribution*, to obtain *the contrastive term distribution*. 
The ROP task is then constructed by sampling pairs of representative word sets from the document.

### Compute Document Term Distribution

Inferencing documents with a max sequence length of 512 take a long time when using only one GPU, so you can specify the `index=i` and `doc_chunk` for running this on multi-GPUs.

We firstly extract the vanilla [CLS]-token attention and store into `h5py` file.

See `./scripts/extract_bert_attention.sh` :

```
DATA_DIR=./data/wiki_info/
Bert_MODEL_DIR=bert-base-uncased-py
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
        --max_seq_length 512 \
        --output_attentions \
        --batch_size 32
```

We then need to transform the vanilla attention weight to distribution by summing up the same words in different position and apply saturation function.

See `./scripts/transform_vanilla_attention_to_term_distribution.sh` :

```
DATA_DIR=./data/wiki_info/bprop

python -m bprop.transform_vanilla_attention_to_term_distribution \
        --train_corpus ${DATA_DIR} \
        --output_dir ${DATA_DIR} \
        --do_lower_case \
        --aggregate sum \
        --saturation k 
```

### Generate representive word sets

The *random term distribution* and the *contrastive term distribution* are computed is this [script](multiprocessing_generate_word_sets.py).

See `./scripts/generate_word_sets.sh` 


```
DATA_DIR=data/wiki_info/bprop

python -m bprop.multiprocessing_generate_word_sets \
    --train_corpus ${DATA_DIR}  \
    --output_dir ${DATA_DIR} \
    --do_lower_case \
    --epochs_to_generate 3 \
    --rop_num_per_doc 10 \
    --method entropy \
    --num_workers 20
```

##  Re-Training BERT with ROP and MLM


See `./scripts/run_pretrain.sh` 

```shell
export DATA_DIR=/path/to/pretraining_data
export Bert_MODEL_DIR=/path/to/pytorch_version/bert-base/
export OUTPUT=/path/to/output

CUDA_VISIBLE_DEVICES="0,1,2,3" \
python run_pretraining.py \
        --pregenerated_data $DATA_DIR \
        --bert_model $Bert_MODEL_DIR \
        --do_lower_case \
        --output_dir $OUTPUT \
        --train_batch_size 80\
        --save_checkpoints_steps 1000\
        --gradient_accumulation_steps 2-5\
        --epochs 1 \
        --negtive_num 1 \
        --learning_rate 2e-5\
        --reduce_memory
```


## Citation
If you find our work useful, please consider citing our paper:
```
@article{ma2020bprop,
  title={B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval},
  author={Ma, Xinyu and Guo, Jiafeng and Zhang, Ruqing and Fan, Yixing and Ji, Xiang and Cheng, Xueqi},
  journal={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```