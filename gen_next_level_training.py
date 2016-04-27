#!/usr/bin/env python

import numpy
from utils import (
    shuffle, train_val_test_split, get_gene_ontology)
import sys
import os
import random
import pandas as pd

LAMBDA = 24
DATA_ROOT = 'data/fofe/'
CUR_LEVEL = 'level_1/'
NEXT_LEVEL = 'level_2/'


go = get_gene_ontology()


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


def get_subtree_set(go_id):
    node = go[go_id]
    if 'go_set' in node:
        return node['go_set']
    go_set = set()
    go_set.add(go_id)
    for ch_id in node['children']:
        if ch_id not in go_set:
            go_set |= get_subtree_set(ch_id)
    node['go_set'] = go_set
    return go_set


def load_data(parent_id, go_id):
    df = pd.read_pickle(DATA_ROOT + NEXT_LEVEL + 'data/' + go_id + '.pkl')
    return df


def main(*args, **kwargs):
    if len(args) < 3:
        raise Exception('Please provide parent id and go id')
    parent_id = args[1]
    go_id = args[2]
    if len(args) == 4:
        level = int(args[3])
        global CUR_LEVEL
        global NEXT_LEVEL
        CUR_LEVEL = 'level_' + str(level) + '/'
        NEXT_LEVEL = 'level_' + str(level + 1) + '/'
    df = load_data(parent_id, go_id)
    go_node = go[go_id]
    for ch_id in go_node['children']:
        ch_set = get_subtree_set(ch_id)
        positives = list()
        negatives = list()
        for i in df.index:
            pos = False
            for g_id in df['gos'][i]:
                if g_id in ch_set:
                    pos = True
                    break
            if pos:
                positives.append(i)
            else:
                negatives.append(i)
        n = min(len(positives), len(negatives))
        if n > 0:
            shuffle(positives)
            shuffle(negatives)
            positives = positives[:n]
            negatives = negatives[:n]
            filename = DATA_ROOT + NEXT_LEVEL + go_id + '/' + ch_id + '.pkl'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            labels = [0] * n + [1] * n
            index = negatives + positives
            new_df = df.reindex(index)
            new_df['labels'] = pd.Series(labels, index=new_df.index)
            new_df.to_pickle(filename)


if __name__ == '__main__':
    main(*sys.argv)
