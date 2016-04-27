#!/usr/bin/env python
import sys
import time
from utils import shuffle, get_gene_ontology
from aaindex import AALETTER
from gensim import corpora, models, matutils
import numpy
from multiprocessing.dummy import Pool as ThreadPool


DATA_ROOT = 'data/prot2vec/'
DICTIONARY_FILE = DATA_ROOT + 'trigram.dict'
CORPUS_FILE = DATA_ROOT + 'corpus.mm'

FILES = (
    'uniparc-all.tab',)

MIN_LEN = 24

INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])

dictionary = corpora.Dictionary.load(DICTIONARY_FILE)


def is_valid(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def create_dictionary():
    words = list()
    for a1 in AALETTER:
        for a2 in AALETTER:
            for a3 in AALETTER:
                words.append(a1 + a2 + a3)
    dictionary = corpora.Dictionary([words])
    # print dictionary.token2id
    dictionary.save(DICTIONARY_FILE)


def prot2doc(seq):
    doc = list()
    ngrams = 3
    for i in range(len(seq) - ngrams):
        doc.append(seq[i:(i + ngrams)])
    return doc


class ProteinSequences(object):
    def __iter__(self):
        with open(DATA_ROOT + 'uniparc-all.tab', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if is_valid(line[1]):
                    yield line[1]


def prot2bow(seq):
    doc = prot2doc(seq)
    return dictionary.doc2bow(doc)


def main():
    start_time = time.time()
    pool = ThreadPool(48)
    print 'Building the corpus'
    corpus = pool.map(prot2bow, ProteinSequences())
    print 'Saving the corpus'
    corpora.MmCorpus.serialize(CORPUS_FILE, corpus)
    # corpus = corpora.MmCorpus(CORPUS_FILE)
    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
