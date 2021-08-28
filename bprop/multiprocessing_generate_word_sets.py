import os
import re
import math
import json
import shelve
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import traceback
import collections
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm, trange
from argparse import ArgumentParser
from multiprocessing import Pool, Lock
from tempfile import TemporaryDirectory
from random import random, randint, shuffle, choice, sample

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize,word_tokenize, PorterStemmer

from pytorch_pretrain_bert.tokenization import BertTokenizer

K = 500 # num of sampling vocabulary
num_instances = 0

lock = Lock()
nltk_stopwords = [l.strip() for l in open('data/stopwords.txt')]
inquery_stopwords = [l.strip() for l in open('data/inquery')]
en_stopwords = set(nltk_stopwords + inquery_stopwords)


class DocumentDatabase:
    def __init__(self, temp_dir='./'):
        self.temp_dir = TemporaryDirectory(dir=temp_dir)
        self.working_dir = Path(self.temp_dir.name)
        self.document_shelf_filepath = self.working_dir / 'shelf.db'
        self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                            flag='n', protocol=-1)
        self.docids = []

    def add_document(self, doc_id, document):
        self.document_shelf[str(doc_id)] = document
        self.docids.append(doc_id)

    def __len__(self):
        return len(self.docids)

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


class Stem2WordDatabase:
    def __init__(self, temp_dir):
        self.temp_dir = TemporaryDirectory(dir=temp_dir)
        self.working_dir = Path(self.temp_dir.name)
        self.document_shelf_filepath = self.working_dir / 'shelf.db'
        self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                            flag='n', protocol=-1,writeback=False)
        self.vocab = []

    def add(self, stemmed_word, word):
        if not stemmed_word:
            return
        if str(stemmed_word) in self.document_shelf:
            if word not in self.document_shelf[str(stemmed_word)]:
                self.document_shelf[str(stemmed_word)][word] = 1
            else:
                self.document_shelf[str(stemmed_word)][word] += 1
        else:
            self.document_shelf[str(stemmed_word)] = {word: 1}
    
    def load(self, stemmed_word2pos):
        for stemmed_word, pos in tqdm(stemmed_word2pos.items()):
            self.vocab.append(str(stemmed_word))
            self.document_shelf[str(stemmed_word)] = pos

    def initialize_prob(self):
        for sw, w_dict in self.document_shelf.items():
            total_num = sum([(math.log(c+1)) for w,c in w_dict.items()])
            w_dict_prob = {}
            for w,c in w_dict.items():
                w_dict_prob[w] = (math.log(c+1))/total_num 
            self.document_shelf[sw] = w_dict_prob

    def sample_word(self, item):
        return np.random.choice(list(self.document_shelf[str(item)]),size=1,p=[v for k,v in self.document_shelf[str(item)].items()])[0]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item):
        return self.document_shelf[str(item)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


class TableForNegativeSamples:
    def __init__(self, vocabs):
        table_size = 1e8
        tables = [np.zeros(int(table_size), dtype=np.uint32) for i in range(len(vocabs))]
        
        # normalize prob
        total_ps = [sum(vocabs[i].values()) for i in range(len(vocabs))]
        
        for layer in range(len(vocabs)):
            p, i = 0, 0
            for j, word in enumerate(vocabs[layer]):
                p += float(vocabs[layer][word]/total_ps[layer]) # normalized prob
                while i < table_size and float(i) / table_size < p:
                    tables[layer][i] = j
                    i += 1
        self.tables = tables
        self.vocabs = [list(vocabs[i]) for i in range(len(vocabs))]

    def sample(self, count, layer=0):
        indices = np.random.randint(low=0, high=len(self.tables[layer]), size=count)
        return [self.vocabs[layer][self.tables[layer][i]] for i in indices]


def softmax(x, t=1):
    """Compute softmax values for each sets of scores in x."""
    x = [v / t for v in x]
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()


def generate_word_sets_from_document(docs, chunk_indexs, stem, rop_num_per_doc, possion_lambda,
                                    vocab_mean_attentions, tokenizer, stem2word, output_filename, method='entropy'):
    for num in chunk_indexs:
        document_data = docs[num]
        document_id = document_data['doc_id']
        document_content = document_data['content'].split()
        document_attentions = document_data['attentions']
        bert_tokenized_content = document_data["bert_tokenized_content"]

        rop_lens = []
        for i in range(rop_num_per_doc):
            while True:
                rop_len = np.random.poisson(possion_lambda)
                if rop_len > 0:
                    rop_lens.append(rop_len)
                    break

        # 80% of the time, we skip short document
        if random() < 0.8 and len(bert_tokenized_content) < 200:
            continue

        doc_vocab_atts = {}
        total_p = sum([a for w, a in document_attentions.items()])
        softmaxed_doc_dis = {w:a/total_p  for w, a in document_attentions.items()}


        for word, att in softmaxed_doc_dis.items():
            # Filtering document vocab, delete stopwords and non-str words
            if not word.isalpha():
                continue
            if method == 'subtract':
                doc_vocab_atts[word] = max(att - vocab_mean_attentions.get(word, 0.), 0)
            elif method == 'entropy':
                doc_vocab_atts[word] = min(att * - math.log(vocab_mean_attentions.get(word, 2.), 2), 100)
            else:
                raise ValueError('Invilid method type! only subtract or entropy.')

        if len(doc_vocab_atts) < 10:
            continue

        document_vocabs = list(doc_vocab_atts)

        # sample_vocabs_score = [{**document_vocab_score[i], **corpus_exclude_document_vocab_score[i]} for i in range(layer_num)]
        sample_vocabs = document_vocabs
        sample_vocabs_score = doc_vocab_atts

        # use softmax instead
        if method == 'subtract':
            T = 0.1 # default 0.1
        elif method == 'entropy':
            T = 1 # default 2
        else:
            raise ValueError('Invilid method type! only subtract or entropy.')

        a  = [v for k, v in sample_vocabs_score.items()]
        # softmax -> sharp
        normalized_prob = softmax(a, T)
        normalized_prob_vocab = {k:normalized_prob[i] for i, (k, v) in enumerate(sample_vocabs_score.items())}        
        sorted_normalized_prob_vocab = {k: v for k, v in sorted(normalized_prob_vocab.items(), key=lambda item: item[1], reverse=True)}

        word_sets = []
        for rop_len in rop_lens:
            word_sets_with_score = []
            for k in range(2):
                replace_flag = False if len(normalized_prob_vocab) > rop_len else True
                rep_words = np.random.choice(list(normalized_prob_vocab),size=rop_len,p=normalized_prob, replace=replace_flag)
                word_sets_score = sum([sample_vocabs_score[w] for w in rep_words])
                word_sets_with_score.append((' '.join(rep_words), word_sets_score))
            word_sets.append(word_sets_with_score)

        instance = {
            'doc_id': document_id,
            "rep_word_sets": word_sets,
            'content_with_softmaxed_prob': sorted_normalized_prob_vocab,
            'content_with_atts': doc_vocab_atts,
            'bert_tokenized_content': bert_tokenized_content
            }

        lock.acquire()
        with open(output_filename,'a+') as epoch_file:
            epoch_file.write(json.dumps(instance, ensure_ascii=False) + '\n')
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
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--method", type=str, default='entropy')
    parser.add_argument("--possion_lambda", type=int, default=3)
    parser.add_argument("--rop_num_per_doc", type=int, default=1,
                        help="Sample n repsentive word sets for each document")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")

    args = parser.parse_args()
    assert os.path.isdir(args.train_corpus)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    doc_idx_pool = []
    vocab_attentions = collections.Counter()
            
    with DocumentDatabase(temp_dir=args.temp_dir) as docs, Stem2WordDatabase(temp_dir=args.temp_dir) as stem2word:
        # file_list = [os.path.join(args.train_corpus, fname) for fname in os.listdir(args.train_corpus) if 'attention' in fname]
        
        # for json_file in file_list:
        json_file = args.train_corpus / f"attention.json"
        print('Processing file: {}'.format(json_file))
        with open(json_file, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Loading Dataset", unit=" lines")):
                # if num > 100:
                #     break
                data = json.loads(line)
                doc_idx = data['doc_id']
                doc_idx_pool.append(doc_idx)
                vocab_attentions.update(data['attentions'])
                docs.add_document(doc_idx, data)

        # Compute the random term distribution
        vocab_mean_attentions = {w:att/len(doc_idx_pool) for w, att in vocab_attentions.items()}
        total_p = sum(vocab_mean_attentions.values())
        normalized_collection_distribution = {w:a/total_p  for w, a in vocab_mean_attentions.items()}

        print('vocab num: {}'.format(len(vocab_mean_attentions)))
        
        if args.stem:
            stem2pos_file = args.train_corpus / f"stem2pos_file.json"
            with open(stem2pos_file, 'r', encoding='utf-8') as f:
                stem2pos = json.load(f)
                stem2word.load(stem2pos)
            stem2word.initialize_prob()
            print('stemmed vocab len:{}, corpus len:{}'.format(len(stem2word), 
                    len(normalized_collection_distribution)))

        # TODO: smoothing for zero probability
        print('Generating word sets...')
        for epoch in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = args.output_dir / f"instances_epoch_{epoch}.json"
            processors = Pool(args.num_workers)
            cand_idxs = doc_idx_pool
            shuffle(cand_idxs)
            for i in range(args.num_workers):
                chunk_size = int(len(cand_idxs) / args.num_workers)
                chunk_indexs = cand_idxs[i*chunk_size:(i+1)*chunk_size]
                res = processors.apply_async(generate_word_sets_from_document, (docs, chunk_indexs, args.stem, args.rop_num_per_doc, \
                args.possion_lambda, normalized_collection_distribution, tokenizer, stem2word, epoch_filename, args.method,), \
                error_callback=error_callback)
            processors.close()
            processors.join()
