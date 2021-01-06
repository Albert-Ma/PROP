# PROP

## Introduction
**PROP**, **P**re-training with **R**epresentative w**O**rds **P**rediction, is a new pre-training method tailored for ad-hoc retrieval. PROP is inspired by the classical statistical language model for IR, specifically the query likelihood model, which assumes that the query is generated as the piece of text representative of the “ideal” document. Based on this idea, we construct the representative words prediction (ROP) task for pre-training. The full paper which describes PROP in detail can be found [here](https://arxiv.org/pdf/2010.10137.pdf).

---
**Note that we pre-trained a different PROP model on MS MARCO document collection using more data and more steps, called [PROP_step400k base](https://drive.google.com/file/d/1aw0s1UK8PvZCI9R8hA9b7kxoN0x35kRr/view?usp=sharing), competing for top positions on the leaderboard. The pre-trained model [PROP_step400k base](https://drive.google.com/file/d/1aw0s1UK8PvZCI9R8hA9b7kxoN0x35kRr/view?usp=sharing) we used for MS MARCO Document Ranking Leaderboard is different from the model in our WSDM 2021 paper.**

* 🔥**News 2021-1-7: PROP_step400k base (ensemble v0.1) got first place on [MS MARCO Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/).**


* 🔥**News 2020-12-16: PROP_step400k base + doc2query top100 (single) tied for 5 place (on test set) on [MS MARCO Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/).**

---

## Pre-training Data

### Download data
For **Wikipedia**, download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) and extract the text with [`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup (e.g. remove spaces and special characters) to convert it into clean text.
For **MS MARCO**, download corpus from the official TREC [website](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz).

### Pre-process data
Compute and store the statistics about `tf, df, average_word_num et.al` on the whole corpus into `Json` file.
* `corpus_df_file.json`: `{word: word document count}`
* `doc_tf_file.json`: `{doc_id, doc_tf, doc_word_num}`, one document per line
* `corpus_tf_file.json`: `{word: word count in the whole corpus}`
* `info_file.json`: `{total_doc_num, total_word_num, average_doc_word_num}`
* `stem2pos_file.json`: `{stem: {word: count}}`
* `preprocessed_data`: `{docid, bert_tokenized_doc_text}`

### Generate representive word sets

```shell
export INPUT_FILE=/path/to/preprocessed_data
export Bert_MODEL_DIR=/path/to/pytorch_version/bert-base/
export OUTPUT=/path/to/output

python multiprocessing_generate_word_sets.py \
    --train_corpus $INPUT_FILE  \
    --do_lower_case \
    --bert_model $Bert_MODEL_DIR \
    --output_dir $OUTPUT \
    --epochs_to_generate 1 \
    --possion_lambda 3 \
    --rop_num_per_doc 10 \
    --num_workers 20 \
    --reduce_memory
```

### Generate training instances

```shell
export INPUT_FILE=/path/to/preprocessed_data
export Bert_MODEL_DIR=/path/to/pytorch_version/bert-base/
export OUTPUT=/path/to/output

python multiprocessing_generate_pairwise_instances.py \
    --train_corpus $INPUT_FILE \
    --bert_model $Bert_MODEL_DIR \
    --do_lower_case \
    --output_dir $OUTPUT \
    --epochs_to_generate 1 \
    --rop_num_per_doc 4 \
    --mlm \
    --max_seq_len 512 \
    --num_workers 20 \
    --reduce_memory
```

## Pre-training


```shell
export DATA_DIR=/path/to/pretraining_data
export Bert_MODEL_DIR=/path/to/pytorch_version/bert-base/
export OUTPUT=/path/to/output

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
## Pre-trained Models
**Download the pre-trained model [PROP_step400k base](https://drive.google.com/file/d/1aw0s1UK8PvZCI9R8hA9b7kxoN0x35kRr/view?usp=sharing) (pre-trained using more data and more steps) we used for MS MARCO Document Leaderboard.**

Download the pre-trained model [PROP<sub>Wikipedia</sub>](), [PROP<sub>MSMARCO</sub>]() we used in our WSDM 2021 paper from Google drive and extract it.

## Fine-tuning
PROP have the same architecture with [`BERT-Base`](https://github.com/google-research/bert), and thus you can fine-tune PROP like BERT on any downstream ad-hoc retrieval tasks by just replacing BERT checkpoints with PROP's. Our clean version of fine-tuning code will be available soon since it contains many irrelevant/WIP code from my main private repository.

## Citation
If you find our work useful, please consider citing our paper:
```
@article{ma2020prop,
  title={PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval},
  author={Ma, Xinyu and Guo, Jiafeng and Zhang, Ruqing and Fan, Yixing and Ji, Xiang and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2010.10137},
  year={2020}
}
```