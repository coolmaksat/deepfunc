#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from utils import get_gene_ontology
from collections import deque


DATA_ROOT = 'data/fofe/'
FILENAME = 'train.txt'


go = get_gene_ontology('go.obo')


def get_go_set(go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set


functions = get_go_set('GO:0003674')


def get_anchestors(go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set


def load_data():
    proteins = list()
    sequences = list()
    gos = list()
    with open(DATA_ROOT + FILENAME, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            proteins.append(items[0])
            sequences.append(items[1])
            go_set = set()
            for go_id in items[2].split('; '):
                if go_id in functions:
                    go_set |= get_anchestors(go_id)
            go_set.remove('GO:0003674')
            gos.append(list(go_set))
    return proteins, sequences, gos


def main(*args, **kwargs):
    proteins, sequences, gos = load_data()
    data = {
        'proteins': np.array(proteins),
        'sequences': np.array(sequences),
        'gos': np.array(gos)}
    df = pd.DataFrame(data)
    df.to_pickle(DATA_ROOT + 'train.pkl')

if __name__ == '__main__':
    main(*sys.argv)
