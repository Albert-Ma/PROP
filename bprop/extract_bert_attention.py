# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import argparse
import collections
import logging
import json
import sys
import re
import numpy as np
from tqdm import tqdm, trange
from random import random

from nltk import sent_tokenize, word_tokenize
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrain_bert.tokenization import BertTokenizer
from pytorch_pretrain_bert.modeling import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, sentence, maps, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.sentence = sentence
        self.maps = maps
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        # tokenize word to sub token and store ori_to_token_map
        truncated_sentences = []
        orig_to_tok_map = []
        bert_tokens = []
        input_type_ids = []
        input_mask = []

        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.append("[CLS]")
        truncated_sentences.append("[CLS]")
        input_type_ids.append(0)
        input_mask.append(1)

        for i, sent in enumerate(sent_tokenize(example.text_a)):
            for j, word in enumerate(word_tokenize(sent)):
                token = tokenizer.tokenize(word)
                if len(token) == 0:
                    continue
                if len(bert_tokens) + len(token) + 1 >= seq_length:
                    break

                truncated_sentences.append(word)
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(token)
                input_type_ids += [0]*len(token)
                input_mask += [1] + [0]*(len(token)-1)

        truncated_sentences.append("[SEP]")
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.append("[SEP]")
        input_type_ids.append(0)
        input_mask.append(1)

        if not example.text_b:
            orig_to_tok_map.append(len(bert_tokens))

        tokens_b = None
        if example.text_b:
            for i, sent in enumerate(sent_tokenize(example.text_b)):
                for j, word in enumerate(word_tokenize(sent)):
                    token = tokenizer.tokenize(word)
                    if len(bert_tokens) + len(token) + 1 >= seq_length:
                        break

                    truncated_sentences.append(word)
                    orig_to_tok_map.append(len(bert_tokens))
                    bert_tokens.extend(token)
                    input_type_ids += [1]*len(token)
                    input_mask += [1] + [0]*(len(token)-1)

            truncated_sentences.append("[SEP]")
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.append("[SEP]")
            input_type_ids.append(1)
            input_mask.append(1)

        if tokens_b:
            if len(bert_tokens) > seq_length:
                raise ValueError("sentence length {} is greater than max_seq_length {}"
                                 .format(len(bert_tokens), seq_length))
        else:
            if len(bert_tokens) > seq_length:
                raise ValueError("sentence length {} is greater than max_seq_length {}"
                                 .format(len(bert_tokens), seq_length))

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("len truncated_sentences: %s" % (len(truncated_sentences)))
            logger.info("len tokens: %s" % (len(bert_tokens)))
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("sentence: %s" % (' '.join(truncated_sentences)))
            logger.info("orig_to_tok_map: %s" % " ".join([str(x) for x in orig_to_tok_map]))
            logger.info("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                sentence=' '.join(truncated_sentences),
                maps=orig_to_tok_map,
                tokens=bert_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file, index=0, doc_chunk=10):
    """Read a list of `InputExample`s from an input file."""
    start_id = index * doc_chunk
    end_id = start_id + doc_chunk - 1 

    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc='Loading doc data...')):
            if i < start_id:
                continue
            if i > end_id:
                break

            if args.corpus_name == 'msmarco':
                if len(line) == 0 or len(line.split('\t'))!=4:
                    continue
                docid, url, title, content = line.split('\t')
                text_content = title + '. ' + content
            elif args.corpus_name == 'wikipedia':
                line = json.loads(line)
                docid = line['id']
                title = line['title']
                content = line['text']
                text_content = title + '. ' + content
            else:
                raise ValueError('error corpus_name!')

            examples.append(
                InputExample(unique_id=docid, text_a=text_content, text_b=None))
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--corpus_name", type=str, required=True, default='wikipedia')
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--index", default=0, type=int, help='Index of collection to start.')
    parser.add_argument("--doc_chunk", default=100000, type=int, help='Num of docs to extract attentions.')
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_embeddings", action='store_true', help="Whether output embeddings.")
    parser.add_argument("--output_attentions", action='store_true', help="Whether output attentions.")
    parser.add_argument("--layers", default="-1", type=str)
    parser.add_argument("--cls", action='store_true', help="Whether only to extract [CLS]'s attention.")
    parser.add_argument("--sub2whole", default='sum', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x)+12 for x in args.layers.split(",")]
    print('layer_indexes: {}'.format(layer_indexes))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    examples = read_examples(args.input_file, args.index, args.doc_chunk)
    print('Would extract {} examples from id-{}!'.format(args.doc_chunk, args.index))

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()
    text_to_index = {}
    with h5py.File(args.output_file, 'w') as fout:
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Generating")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, example_indices = batch

            all_encoder_layers = model(input_ids, token_type_ids=None, attention_mask=input_mask, output_attentions=args.output_attentions)
            if args.output_attentions:
                attentions_outputs = all_encoder_layers[2]
            all_encoder_layers = all_encoder_layers[0]

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = feature.unique_id
                text_to_index[feature.sentence] = str(unique_id)
                orig_to_tok_map = feature.maps

                embeddings = np.zeros((len(layer_indexes), len(feature.sentence.split()), 768), dtype=float)
                attentions = np.zeros((len(layer_indexes), len(feature.sentence.split()), 12, args.max_seq_length), dtype=float)
                
                # SUM sub_token embedding to word embedding(word_emb)
                for i in range(len(orig_to_tok_map)-1):
                    # Only extract [CLS] attention
                    if args.cls and i != 0:
                        break
                    
                    tmp_emb = [[] for i in range(len(layer_indexes))]
                    tmp_att = [[] for i in range(len(layer_indexes))]

                    for j in range(orig_to_tok_map[i], orig_to_tok_map[i+1]):
                        for (k, layer_index) in enumerate(layer_indexes):
                            # embeddings
                            if args.output_embeddings:
                                layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                                layer_output = layer_output[b]
                                values = [round(x.item(), 6) for x in layer_output[j]]
                                tmp_emb[k].append(values)

                            # attentions
                            # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, seq_len, num_heads, seq_len)
                            if args.output_attentions: # Only extract [CLS] attention
                                attention_output = attentions_outputs[int(layer_index)].detach().cpu().numpy()
                                attention_output = np.transpose(attention_output[b],(1,0,2))
                                tmp_att[k].append(attention_output[j].squeeze())

                    if args.output_embeddings:
                        if args.sub2whole == 'sum':
                            for (k, layer_index) in enumerate(layer_indexes):
                                whole_word_emb = np.zeros(768)
                                for v in tmp_emb[k]:
                                    whole_word_emb += np.array(v)
                                embeddings[k, i] = whole_word_emb
                        elif args.sub2whole == 'average':
                            for (k, layer_index) in enumerate(layer_indexes):
                                whole_word_emb = np.zeros(768)
                                for v in tmp_emb[k]:
                                    whole_word_emb += np.array(v)
                                embeddings[k, i] = whole_word_emb/len(tmp_emb[k])
                        elif args.sub2whole == 'first':
                            for (k, layer_index) in enumerate(layer_indexes):
                                embeddings[k, i] = np.array(tmp_emb[k][0])
                        else:
                            raise ValueError('Invalid sub2whole param!')

                    if args.output_attentions:
                        if args.sub2whole == 'sum':
                            for (k, layer_index) in enumerate(layer_indexes):
                                whole_word_attentions = np.zeros((12, args.max_seq_length))
                                for v in tmp_att[k]:
                                    whole_word_attentions += np.array(v)
                                attentions[k, i] = whole_word_attentions
                        if args.sub2whole == 'first':
                            # Only extract cls
                            for (k, layer_index) in enumerate(layer_indexes):
                                whole_word_attentions = np.zeros((12, args.max_seq_length))
                                for index, start_index in enumerate(orig_to_tok_map):
                                    whole_word_attentions[:,index] = tmp_att[k][0][:,start_index] # (seq_len, num_heads, seq_len)
                                attentions[k, i] = whole_word_attentions

                if args.output_embeddings:
                    if len(layer_indexes) == 1:
                        out = embeddings[-1]
                    else:
                        out = embeddings
                    
                    fout.create_dataset(
                        str(unique_id),
                        out.shape, dtype='float32',
                        data=out, compression="gzip", compression_opts=9)

                if args.output_attentions:
                    if len(layer_indexes) == 1:
                        att = attentions[-1]
                    else:
                        att = attentions

                    fout.create_dataset(
                        str(unique_id)+"_attention",
                        att.shape, dtype='float32',
                        data=att, compression="gzip", compression_opts=9)
        text_index_dataset = fout.create_dataset(
            "text_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str), compression="gzip", compression_opts=9)
        text_index_dataset[0] = json.dumps(text_to_index)

