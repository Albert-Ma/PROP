import os
import re
import math
import json
import time
import shelve
import traceback
import collections
import numpy as np
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


K = 500 
TEMP_DIR = './'
num_instances = 0

lock = Lock()
nltk_stemmer = PorterStemmer()
nltk_tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')


class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.reduce_memory = reduce_memory

    def add_document(self, doc_idx, document):
        if not document:
            return
        if self.reduce_memory:
            self.document_shelf[str(doc_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

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

class DocumentTfDatabase:
    def __init__(self):
        self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
        self.working_dir = Path(self.temp_dir.name)
        self.document_shelf_filepath = self.working_dir / 'shelf.db'
        self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                            flag='n', protocol=-1)
        self.doc_ids = []

    def add_document_tf(self, doc_id, document_tf):
        if not document_tf:
            return
        self.document_shelf[str(doc_id)] = document_tf
        self.doc_ids.append(doc_id)

    def __len__(self):
        return len(self.doc_ids)

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

    def exit(self):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

class Stem2WordDatabase:
    def __init__(self):
        self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
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
    
    def exit(self):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

class TableForNegativeSamples:
    def __init__(self, vocab):
        table_size = 1e8
        table = np.zeros(int(table_size), dtype=np.uint32)

        p, i = 0, 0
        for j, word in enumerate(vocab):
            p += float(vocab[word]) # normalized df
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
        self.vocab = list(vocab)

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.vocab[self.table[i]] for i in indices]

def generate_word_sets_from_document(docs, chunk_indexs, stem, rop_num_per_doc, possion_lambda, corpus_vocab_with_idf, docs_tf, normalized_df,
                                    negative_table, average_doc_word_num, word_discards, stem2word, sample_prob, epoch_file_name):
    for num in chunk_indexs:
        document_data = docs[num]
        document_id = document_data['id']
        bert_tokenized_document = document_data['bert_tokenized_content']

        if document_id not in docs_tf:
            continue

        doc_tf_data = docs_tf[document_id] # {'id': , 'tf': , 'word_num':len(doc_words_no_punctualtion)}
        doc_tf = doc_tf_data['tf']
        doc_word_num = doc_tf_data['word_num']

        rop_lens = []
        for i in range(rop_num_per_doc):
            while True:
                rop_len = np.random.poisson(possion_lambda)
                if rop_len > 0:
                    rop_lens.append(rop_len)
                    break

        # 80% of the time, we skip short document
        if random() < 0.8 and len(pos_bert_tokenized_doc) < 200:
            continue

        # For computational efficiency, we use top-k sampling from corpus vocabulary.
        # Specially, we sample words which are not occured given a document, and then
        # form the final sampling vocabulary with the words in the document.
        # Another reason is that, according to the unigram document language model with
        # dirichlet smoothing function, most words (which is not occured in a specific
        # document) remain the same probality in most time.
        document_vocab_score = {}
        document_vocab = list(doc_tf)
        for w in document_vocab:
            tf = doc_tf[w] if w in doc_tf else 0
            df = normalized_df[w] if w in normalized_df else 0
            if sample_prob == 'ql' or sample_prob == 'random':
                document_vocab_score[w] = (tf + average_doc_word_num*df) / (doc_word_num + average_doc_word_num)
            else:
                raise ValueError('sample prob({}) is invalid!'.format(sampe_prob))

        corpus_exclude_document_vocab_score = {}
        corpus_sample_vocab = negative_table.sample(K-len(document_vocab))
        for w in corpus_sample_vocab:
            if w in document_vocab:
                continue
            df = normalized_df[w] if w in normalized_df else 0
            corpus_exclude_document_vocab_score[w] = (0 + average_doc_word_num*df) / (doc_word_num + average_doc_word_num)

        sample_vocab = document_vocab + corpus_sample_vocab
        sample_vocab_score = dict(document_vocab_score.items() + corpus_exclude_document_vocab_score.items())

        normalized_prob_vocab = {}
        total_p = sum([v for k,v in sample_vocab_score.items()])
        for k, v in sample_vocab_score.items():
            normalized_prob_vocab[k] = v/total_p

        word_sets = []
        for rop_len in rop_lens:
            rep_words = []
            word_sets_with_score = []
            for k in range(2):
                while len(rep_words) < rop_len:
                    wd = np.random.choice(list(normalized_prob_vocab),size=1,p=[v for k,v in normalized_prob_vocab.items()])[0]
                    if word_discards[wd] < np.random.rand():
                        continue
                    else:
                        rep_words.append(wd)
                word_sets_score = sum([math.log(sample_vocab_score[w]) for w in rep_words])
                if stem:
                    rep_words = [stem2word.sample_word(w) for w in rep_words]
                word_sets_with_score.append((' '.join(rep_words), word_sets_score))
            word_sets.append(word_sets_with_score)

        instance = {
            "rep_word_sets": word_sets,
            'bert_tokenized_content': bert_tokenized_document
            }

        lock.acquire()
        with open(epoch_filename,'a+') as epoch_file:
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
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--possion_lambda", type=int, default=3)
    parser.add_argument("--rop_num_per_doc", type=int, default=1,
                        help="Sample n repsentive word sets for each document")
    parser.add_argument("--sample_prob", type=str, default='ql')
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    args = parser.parse_args()

    assert os.path.isdir(args.train_corpus)

    doc_idx_pool = []
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        docs_file = args.train_corpus / f"docs.json"
        with docs_file.open() as f:
            for i, line in enumerate(tqdm(f, desc="Loading Dataset", unit=" lines")):
                data = json.loads(line)
                doc_idx = data['id']
                doc_idx_pool.append(doc_idx)
                docs.add_document(doc_idx, data)

            if len(docs) <= 1:
                exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                    "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                    "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                    "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                    "sections or paragraphs.")
        print('Reading file is done! Total doc num:{}'.format(len(docs)))

        args.output_dir.mkdir(exist_ok=True)
        corpus_df_file = args.train_corpus / f"corpus_df_file.json"
        doc_tf_file = args.train_corpus / f"doc_tf_file.json"
        corpus_tf_file = args.train_corpus / f"corpus_tf_file.json"
        corpus_info_file = args.train_corpus / f"info_file.json"
        stem2pos_file = args.train_corpus / f"stem2pos_file.json"
        
        corpus_word_df = {} # count
        doc_word_tf = {} # count
        corpus_word_tf = {} # count
        stem2pos = {} # count
        total_word_num = 0
        average_doc_word_num = 0
        stem2word = Stem2WordDatabase()
        
        if os.path.isfile(corpus_info_file) and os.path.isfile(corpus_tf_file) \
            and os.path.isfile(doc_tf_file)  and os.path.isfile(corpus_df_file) :
            docs_tf = DocumentTfDatabase()
            with open(corpus_info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_doc_num = data["total_doc_num"]
                total_word_num = data['total_word_num']
                average_doc_word_num = data["average_doc_word_num"]
        
            print('loading tf df info from file...')
            with open(doc_tf_file, 'r', encoding='utf-8') as f:    
                for line in tqdm(f, desc='Loading doc tf data...'):
                    data = json.loads(line)
                    doc_idx = data['id']
                    docs_tf.add_document_tf(doc_idx, data)

            print('Loading tf-df info is done! Total doc num:{}'.format(len(docs_tf)))

            with open(corpus_tf_file, 'r', encoding='utf-8') as f:
                corpus_word_tf = json.load(f)

            with open(corpus_df_file, 'r', encoding='utf-8') as f:
                corpus_word_df = json.load(f)

            if args.stem:
                with open(stem2pos_file, 'r', encoding='utf-8') as f:
                    stem2pos = json.load(f)
                    stem2word.load(stem2pos)
                stem2word.initialize_prob()
                print('stemmed vocab len:{}, corpus len:{}'.format(len(stem2word), len(corpus_word_df)))

            total_df = 0
            corpus_vocab = {}
            normalized_df = {}
            for k, v in corpus_word_df.items():
                if corpus_word_tf[k] < 100:
                    continue
                normalized_df[k] = v/total_doc_num
                total_df += normalized_df[k]
                corpus_vocab[k] = math.log(total_doc_num/(v+1))
            for k,v in normalized_df.items():
                normalized_df[k] = v/total_df
            
            print('total_doc_num:{}, average_doc_word_num:{},corpus_vocab num:{}'.format(
                total_doc_num, average_doc_word_num, len(corpus_vocab)))
            
            negative_table = TableForNegativeSamples(normalized_df)

            # The subsampling methods will randomly discards frequent words 
            word_discards = {}
            t = 1e-4 # 1e-5
            for k,v in corpus_word_tf.items():
                # Ref to https://github.com/tmikolov/word2vec/blob/master/word2vec.c#L409
                if corpus_word_tf[k] < 100:
                    continue
                word_fre = v/total_word_num
                discards = math.sqrt(t / word_fre) + (t / word_fre)
                word_discards[k] = discards

            del corpus_word_tf, corpus_word_df, stem2pos

            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                epoch_filename = args.output_dir / f"instances_{epoch}.json"
                num_processors = args.num_workers
                processors = Pool(num_processors)
                cand_idxs = doc_idx_pool
                shuffle(cand_idxs)
                for i in range(num_processors):
                    chunk_size = int(len(cand_idxs) / num_processors)
                    chunk_indexs = cand_idxs[i*chunk_size:(i+1)*chunk_size]
                    res = processors.apply_async(generate_word_sets_from_document, (docs, chunk_indexs, args.stem, args.rop_num_per_doc, args.possion_lambda,\
                    corpus_vocab, docs_tf, normalized_df, negative_table, average_doc_word_num, word_discards, stem2word, args.sample_prob, epoch_filename,), \
                    error_callback=error_callback)
                processors.close()
                processors.join()
    
    docs_tf.exit()
    if args.stem:
        stem2word.exit()