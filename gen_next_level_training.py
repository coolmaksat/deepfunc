#!/usr/bin/env python

import numpy
from utils import (
    shuffle, train_val_test_split, get_gene_ontology, get_model_max_features)
import sys
import os

LAMBDA = 24
DATA_ROOT = 'data/swiss/'

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


def load_data_by_prot_id(go_id):
    data = dict()
    with open(DATA_ROOT + 'level_2/data/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split()
            prot_id = line[0]
            data[prot_id] = line[1:]
    return data


def load_training_data(go_id):
    go_set = get_subtree_set(go_id)
    data = list()
    with open(DATA_ROOT + 'train.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            gos = line[2].split('; ')
            ok = False
            go_ids = list()
            for go_id in gos:
                if go_id in go_set:
                    ok = True
                    go_ids.append(go_id)
            if ok:
                data.append((prot_id, go_ids))
    return data


def main(*args, **kwargs):
    if len(args) != 2:
        raise Exception('Please provide function id')
    go_id = args[1]
    paacs = load_data_by_prot_id(go_id)
    data = load_training_data(go_id)
    go_node = go[go_id]
    go_set = get_subtree_set(go_id)
    for ch_id in go_node['children']:
        ch_set = get_subtree_set(ch_id)
        positives = list()
        negatives = list()
        for prot_id, gos in data:
            if prot_id not in paacs:
                continue
            pos = False
            for g_id in gos:
                if g_id in ch_set:
                    pos = True
                    break
            if pos:
                positives.append(prot_id)
            else:
                negatives.append(prot_id)
        n = len(positives)
        shuffle(positives)
        shuffle(negatives)
        negatives = negatives[:n]
        with open(DATA_ROOT + 'level_2/' + go_id + '/' + ch_id + '.txt', 'w') as f:
            for prot_id in negatives:
                f.write('0 ' + prot_id)
                for p in paacs[prot_id]:
                    f.write(' ' + str(p))
                f.write('\n')
            for prot_id in positives:
                f.write('1 ' + prot_id)
                for p in paacs[prot_id]:
                    f.write(' ' + str(p))
                f.write('\n')


if __name__ == '__main__':
    main(*sys.argv)
