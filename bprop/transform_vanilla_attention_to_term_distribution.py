import os
import json
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import collections
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from argparse import ArgumentParser
from nltk import PorterStemmer

from pytorch_pretrain_bert.tokenization import BertTokenizer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--aggregate", type=str, default='sum')
    parser.add_argument("--saturation", type=str, default='k')
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    print('**** preprocessed output dir:{}'.format(args.output_dir))
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    attention_file = args.output_dir / f"attention.json"
    output_attention_file = open(attention_file,'a+')

    stem2pos = {}
    stemmer = PorterStemmer()
    stem2pos_file = args.output_dir / f"stem2pos_file.json"


    file_list = [fname for fname in os.listdir(args.train_corpus) if '.hdf5' in fname]
    # file_list = [args.train_corpus]
    for h5py_file in file_list:
        h5py_file = args.train_corpus / h5py_file
        print('Processing file: {}'.format(h5py_file))
        with h5py.File(h5py_file, "r") as attentions:
            text_to_index = json.loads(attentions.get("text_to_index")[0])
            for text, index in tqdm(text_to_index.items(), leave=False):
                data = {}
                data['doc_id'] = index
                # Lowercase text
                data['bert_tokenized_content'] = bert_tokenizer.tokenize(' '.join(text.split()))
                if args.do_lower_case:
                    text = text.lower()
                data['content'] = text

                assert len(data['bert_tokenized_content']) <= 512

                atts = attentions[index+"_attention"][:]
                # (word, head_num, max_seq_length) -> #  cls's attention (max_seq_length)
                mean_attens = np.mean(atts, axis=1) # (word, max_seq_length)

                # extract cls's mean attention to other words
                mean_atts_with_words = [(w, mean_attens[0][w_index]) for w_index, w in enumerate(text.split())]

                doc2att = {}
                for word, att in mean_atts_with_words:
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

                    if args.aggregate == 'max':
                        doc2att[word] = max(att, doc2att.get(word, 0))
                    elif args.aggregate == 'sum':
                        doc2att[word] = doc2att.get(word, 0.) + att
                    else:
                        raise ValueError('Invilid Aggregate type! only max or sum.')
                
                for k, v in doc2att.items():
                    if args.saturation == 'sqrt':
                        doc2att[k] = math.sqrt(v)
                    elif args.saturation == 'k':
                        # default
                        doc2att[k] = v/(v+0.01)
                    elif args.saturation == 'e':
                        doc2att[k] = 1/(1+math.exp(-(v-0.02)*200))

                data['attentions'] = doc2att
                output_attention_file.write(json.dumps(data, ensure_ascii=False) + '\n')
        print('Processing file: {} done!'.format(h5py_file))
    if args.stem:
            with stem2pos_file.open('w', encoding='utf-8') as f:
                f.write(json.dumps(stem2pos, ensure_ascii=False))
