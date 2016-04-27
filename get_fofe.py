#!/usr/bin/env python
import sys
import numpy
import itertools
import pandas as pd


AA = 'ARNDCEQGHILKMFPSTWYV'
ALPHA = 0.999
NGRAM = 3
DATA_ROOT = 'data/fofe/'
FILENAME = 'test.txt'


def load_data():
    docs = list()
    proteins = list()
    with open(DATA_ROOT + FILENAME, 'r') as f:
        for line in f:
            items = line.split('\t')
            prot_id = items[0]
            seq = items[1]
            docs.append(seq)
            proteins.append(prot_id)
    return proteins, docs


def convert_data(docs):
    bgram = [''.join(item) for item in itertools.product(AA, repeat=NGRAM)]
    bgram_index = dict()
    for i, s in enumerate(bgram):
        bgram_index[s] = i

    X = numpy.zeros((len(docs), len(bgram)))
    for doc in range(0, len(docs)):
        d = docs[doc][:-1]
        seq = [d[i:(i + NGRAM)] for i in range(len(d) - NGRAM + 1)]
        for word in seq:
            X[doc, :] *= ALPHA
            col_i = bgram_index[word]
            X[doc, col_i] += 1
    return X


def main(*args, **kwargs):
    proteins, docs = load_data()
    X = convert_data(docs)
    data = {
        'proteins': numpy.array(proteins), 'data': list(X), 'sequence': docs}
    df = pd.DataFrame(data)
    df.to_pickle(DATA_ROOT + 'test.pkl')

if __name__ == '__main__':
    main(*sys.argv)
