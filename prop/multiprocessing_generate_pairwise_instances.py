import re
import os
import json
import shelve
import traceback
import collections
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from multiprocessing import Pool, Value, Lock
from random import random, shuffle, choice, sample, randrange

import numpy as np
from tqdm import tqdm

from pytorch_pretrain_bert.tokenization import BertTokenizer

lock = Lock()
num_instances = Value('i', 0)

class DocumentDatabase:
    def __init__(self, temp_dir='./'):
        self.temp_dir = TemporaryDirectory(dir=temp_dir)
        self.working_dir = Path(self.temp_dir.name)
        self.document_shelf_filepath = self.working_dir / 'shelf.db'
        self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                            flag='n', protocol=-1)
        self.indexes = []

    def add_document(self, document):
        current_idx = len(self.indexes)
        self.document_shelf[str(current_idx)] = document
        self.indexes.append(current_idx)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        return self.document_shelf[str(item)]

    def __contains__(self, item):
        if str(item) in self.document_shelf:
            return True
        else:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        # truncate from the doc side
        tokens_b.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    # [MASK] word from DOC, not the query
    START_DOC = False
    for (i, token) in enumerate(tokens):
        if token == "[SEP]":
            START_DOC = True
            continue
        if token == "[CLS]":
            continue
        if not START_DOC:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(cand_indices) * masked_lm_prob))))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels

def construct_pairwise_examples(docs, chunk_indexs, rop_num_per_doc, max_seq_len, mlm, 
                                bert_tokenizer, masked_lm_prob, max_predictions_per_seq,
                                bert_vocab_list, epoch_filename):
    for idx in chunk_indexs:
        example = docs[idx]
        rep_word_sets = example["rep_word_sets"]
        pos_bert_tokenized_doc = example['bert_tokenized_content']
        assert rop_num_per_doc <= len(rep_word_sets)

        # Sample $M$ pair of candiate word sets
        rep_word_sets_indexes = list(range(len(rep_word_sets)))
        cand_rep_word_sets_indexes = np.random.choice(rep_word_sets_indexes, rop_num_per_doc, replace=False)
        
        # Construct positive and negative examples
        instances = []
        for i in range(len(rep_word_sets)):
            if i not in cand_rep_word_sets_indexes:
                continue
            pairwise_rep_word_sets = rep_word_sets[i]
            rep_word_set1, rep_word_set1_score = pairwise_rep_word_sets[0]
            rep_word_set2, rep_word_set2_score = pairwise_rep_word_sets[1]

            if rep_word_set1_score > rep_word_set2_score:
                pos_rep_word_set = rep_word_set1
                neg_rep_word_set = rep_word_set2
            else:
                pos_rep_word_set = rep_word_set2
                neg_rep_word_set = rep_word_set1

            for j, word_sets in enumerate([pos_rep_word_set, neg_rep_word_set]):
                query_tokens = bert_tokenizer.tokenize(word_sets)
                doc_tokens = pos_bert_tokenized_doc
                assert len(query_tokens) >= 1 and len(doc_tokens) >= 1

                label = 1 if j == 0 else 0

                truncate_seq_pair(query_tokens, doc_tokens, max_seq_len-3)
                tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + doc_tokens + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(query_tokens) + 2)] + [1 for _ in range(len(doc_tokens) + 1)]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)

                input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
                masked_label_ids = bert_tokenizer.convert_tokens_to_ids(masked_lm_labels)

                instance = {
                    "input_ids": input_ids,
                    "segment_ids": segment_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_label_ids": masked_label_ids,
                    "label": label,
                    }
                instances.append(instance)

        lock.acquire()
        with open(epoch_filename,'a+') as epoch_file:
            for instance in instances:
                epoch_file.write(json.dumps(instance, ensure_ascii=False) + '\n')
                num_instances.value += 1
        lock.release()

def error_callback(e):
    print('error')
    print(dir(e), "\n")
    traceback.print_exception(type(e), e, e.__traceback__)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--temp_dir", type=str, default='./')
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--rop_num_per_doc", type=int, default=1,
                        help="How many samples for each document")
    parser.add_argument("--epochs_to_generate", type=int, default=1,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--mlm", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=60,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    bert_vocab_list = list(bert_tokenizer.vocab.keys())

    for epoch in range(args.epochs_to_generate):
        epoch_data = args.train_corpus / f"instances_epoch_{epoch}.json"
        with DocumentDatabase(temp_dir=args.temp_dir) as docs:
            with epoch_data.open() as f:
                for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                    example = json.loads(line)
                    docs.add_document(example)
                if len(docs) <= 1:
                    exit("ERROR: No document breaks were found in the input file!")
            print('Reading file is done! Total doc num:{}'.format(len(docs)))

            instances = []
            num_instances.value = 0
            output_epoch_filename = args.output_dir / f"epoch_{epoch}.json"
            processors = Pool(args.num_workers)
            cand_idxs = list(range(0, len(docs)))
            shuffle(cand_idxs)
            for i in range(args.num_workers):
                chunk_size = int(len(cand_idxs) / args.num_workers)
                chunk_indexs = cand_idxs[i*chunk_size:(i+1)*chunk_size]
                r = processors.apply_async(construct_pairwise_examples, (docs, chunk_indexs, args.rop_num_per_doc, args.max_seq_len, \
                    args.mlm, bert_tokenizer, args.masked_lm_prob, args.max_predictions_per_seq, bert_vocab_list, output_epoch_filename,),\
                    error_callback=error_callback)
            processors.close()
            processors.join()

            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances.value,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))       