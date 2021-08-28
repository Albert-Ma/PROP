import re
import json
import shelve
import collections
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize, word_tokenize, PorterStemmer

from pytorch_pretrain_bert.tokenization import BertTokenizer


nltk_stopwords = [l.strip() for l in open('data/stopwords.txt')]
inquery_stopwords = [l.strip() for l in open('data/inquery')]
en_stopwords = set(nltk_stopwords + inquery_stopwords)

stemmer = PorterStemmer()
nltk_tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # numbers? [0-9]


class DocumentDatabase:
    def __init__(self, temp_dir):
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--corpus_name", type=str, required=True, default='wikipedia')
    parser.add_argument('--data_file', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--temp_dir", type=str, default='./')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--stem", action="store_true")
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True)
    print('**** preprocessed output dir:{}'.format(args.output_dir))

    bert_tokenized_docs = args.output_dir / f"bert_tokenized_docs.json"
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    doc_idx_pool = []
    with DocumentDatabase(args.temp_dir) as docs, open(bert_tokenized_docs,'w') as fin:
        with args.data_file.open(encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
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

                if args.do_lower_case:
                    text_content = text_content.lower()
                tokens = nltk_tokenizer.tokenize(text_content)
                doc_idx_pool.append(docid)
                docs.add_document(docid, tokens)

                bert_tokenized_document = bert_tokenizer.tokenize(' '.join(text_content.split()[:512]))
                doc_instance = {'id': docid, 'contents': bert_tokenized_document}
                fin.write(json.dumps(doc_instance, ensure_ascii=False) + '\n')


            if len(docs) <= 1:
                exit("ERROR: No documents were found in the input file!")
        print('Reading file is done! Total doc num:{}'.format(len(docs)))

        corpus_df_file = args.output_dir / f"corpus_df_file.json"
        corpus_tf_file = args.output_dir / f"corpus_tf_file.json"
        doc_tf_file = args.output_dir / f"doc_tf_file.json"
        corpus_info_file = args.output_dir / f"info_file.json"
        stem2pos_file = args.output_dir / f"stem2pos_file.json"
        
        corpus_word_df, corpus_word_tf, stem2pos = {}, {}, {} # count
        total_word_num, average_doc_word_num = 0, 0

        with open(doc_tf_file, 'w', encoding='utf-8') as tf_file:
            for id_index in trange(len(doc_idx_pool), desc="Computing tf-df info..."):
                doc_id = doc_idx_pool[id_index]
                doc_tokens = docs[doc_id]
                tokens = [t for t in doc_tokens if t.lower() not in en_stopwords]

                doc_word_tf = {}
                doc_word_set = set()
                for word in tokens:
                    if args.stem:
                        stemmed_word = stemmer.stem(word)
                        # stemmed word to its diff pos words
                        if stemmed_word in stem2pos:
                            if word not in stem2pos[stemmed_word]:
                                stem2pos[stemmed_word][word] = 1
                            else:
                                stem2pos[stemmed_word][word] += 1
                        else:
                            stem2pos[stemmed_word] = {word: 1}
                        word = stemmed_word
                    
                    doc_word_set.add(word)

                    # tf in doc
                    if word in doc_word_tf:
                        doc_word_tf[word] += 1
                    else:
                        doc_word_tf[word] = 1

                    # tf in corpus
                    if word in corpus_word_tf:
                        corpus_word_tf[word] += 1
                    else:
                        corpus_word_tf[word] = 1

                doc_tf = {'id': doc_id, 
                          'tf': doc_word_tf,
                          'word_num':len(tokens)
                        }
                tf_file.write(json.dumps(doc_tf, ensure_ascii=False) + '\n')

                # df
                for word in doc_word_set:
                    if word in corpus_word_df:
                        corpus_word_df[word] += 1
                    else:
                        corpus_word_df[word] = 1

                total_word_num += len(tokens)

        total_doc_num = len(docs)
        average_doc_word_num = total_word_num/total_doc_num

        with corpus_info_file.open('w', encoding='utf-8') as f:
            data = {
                    "total_doc_num": total_doc_num,
                    "total_word_num": total_word_num, 
                    "average_doc_word_num": total_word_num/total_doc_num
                }
            f.write(json.dumps(data, ensure_ascii=False))

        with corpus_tf_file.open('w', encoding='utf-8') as f:
            f.write(json.dumps(corpus_word_tf, ensure_ascii=False))

        with corpus_df_file.open('w', encoding='utf-8') as f:
            f.write(json.dumps(corpus_word_df, ensure_ascii=False))

        if args.stem:
            with stem2pos_file.open('w', encoding='utf-8') as f:
                f.write(json.dumps(stem2pos, ensure_ascii=False))