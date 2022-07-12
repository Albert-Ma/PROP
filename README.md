# PROP


## Introduction
**PROP**, **P**re-training with **R**epresentative w**O**rds **P**rediction, is a new pre-training method tailored for ad-hoc retrieval. PROP is inspired by the classical statistical language model for IR, specifically the query likelihood model, which assumes that the query is generated as the piece of text representative of the “ideal” document. Based on this idea, we construct the representative words prediction (ROP) task for pre-training. The full paper can be found [here](https://arxiv.org/pdf/2010.10137.pdf).

---

* 2022-7 Update: we have uploaded the PROP model to the huggingface model cards.
* 2021-8 Update: we add the [B-PROP](bprop/README.md) pre-training in this repo.
* 🔥**News 2021-1-7: PROP_step400k base (ensemble v0.1) got first place on [MS MARCO Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/).**


---


## Pre-trained models in the Huggingface

We have uploaded three PROP models to the Huggingface Hub, so you can easily use the PROP models with [Huggingface/Transformers](https://github.com/huggingface/transformers) library.

Model identifier in Huggingface Hub:

- `xyma/PROP-wiki`: The official PROP model pre-trained on Wikipedia and used in our WSDM'2021 paper
- `xyma/PROP-marco`: The official PROP model pre-trained on MS MARCO document corpus and used in our WSDM'2021 paper
- `xyma/PROP-marco-step400k`: PROP model pre-trained on MS MARCO document corpus with more steps and used in MS MARCO Document Ranking Leaderboard

For example,
```
tokenizer = AutoTokenizer.from_pretrained("xyma/PROP-marco")
model = AutoModel.from_pretrained("xyma/PROP-marco")
```

> Note that if you want to test the zero-shot performance of PROP models, please use this repo or create a PROP model like [this](https://github.com/Albert-Ma/PROP/blob/main/pytorch_pretrain_bert/modeling.py#L775) since we use a different CLS head with BERT.
The CLS head identifier in PROP model weights is only `cls` which is a linear module (768, 1).

For example, you can ceate PROP class like the following before fine-tuning,
```
from transformers import BertPreTrainedModel, BertModel, BertLMPredictionHead

class PROP(BertPreTrainedModel):
  def __init__(self, config):
      super(PROP, self).__init__(config)
      self.bert = BertModel(config)
      self.cls = torch.nn.Linear(config.hidden_size, 1)
      self.cls.predictions = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
      self.init_weights()
  ...
```

Also you can download the PROP models from the Google drive:

- `PROP-wiki`: [download link](https://drive.google.com/file/d/11uj30VgEsVLj6PekP-SBvjWzlyLDP7Hf/view?usp=sharing)
- `PROP-marco`: [download link](https://drive.google.com/file/d/1E2E-kS_gXp28imhjNdNGNX8EFGt5cs-5/view?usp=sharing)
- `PROP-marco-step400k`: [download link](https://drive.google.com/file/d/1aw0s1UK8PvZCI9R8hA9b7kxoN0x35kRr/view?usp=sharing)



## Pre-training Corpus

### Download data
For **Wikipedia**, download [the dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) and extract the text with [`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup (e.g. remove spaces and special characters) to convert it into clean text.
For **MS MARCO**, download corpus from the official TREC [website](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz).

### Pre-process data
Compute and store the statistics about `tf, df, average_word_num et.al` on the whole corpus into `Json` file. Also, you can use the IR toolkit like lucene to get these info.

Run `./scripts/preprocess.sh` to generate the following files:

* `corpus_df_file.json`: `{word: document tf}`
* `doc_tf_file.json`: `{doc_id, doc_tf, doc_word_num}`, one document per line
* `corpus_tf_file.json`: `{word: corpus tf}`
* `info_file.json`: `{total_doc_num, total_word_num, average_doc_word_num}`
* `stem2pos_file.json`: `{stem: {word: count}}`
* `preprocessed_data`: `{docid, bert_tokenized_doc_text}`

### Generate representive word sets

See `./scripts/generate_word_sets.sh` 

The details are as follows and you can tune the hyperparameter `possion_lambda` to match with your target dataset (the average of query length):

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

See `./scripts/generate_pair_instances.sh` 

The details are as follows and other hyperparameters are set to the default value:

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



## Fine-tuning
PROP have the same architecture with [`BERT-Base`](https://github.com/google-research/bert), and thus you can fine-tune PROP like BERT on any downstream ad-hoc retrieval tasks by just replacing BERT checkpoints with PROP's.

```
class ReRanker(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=1)
        self.prop = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, group_num=2):
        batch_size = input_ids.size(0)
        output  = self.prop(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, return_dict=True)
        logits = output['logits']

        if labels is not None:
            logits = logits.reshape(batch_size//group_num, group_num)
            pairwise_labels = torch.zeros(batch_size//group_num, dtype=torch.long).to(logits.device)
            pairwise_loss = self.cross_entropy(logits, pairwise_labels)
            return pairwise_loss
        else:
            return logits
```


## Citation
If you find our work useful, please consider citing our paper:
```
@article{ma2020prop,
  title={PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval},
  author={Ma, Xinyu and Guo, Jiafeng and Zhang, Ruqing and Fan, Yixing and Ji, Xiang and Cheng, Xueqi},
  journal={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  year={2021}
}

@article{ma2020bprop,
  title={B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval},
  author={Ma, Xinyu and Guo, Jiafeng and Zhang, Ruqing and Fan, Yixing and Ji, Xiang and Cheng, Xueqi},
  journal={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```