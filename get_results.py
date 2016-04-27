#!/usr/bin/env python
import sys
from collections import deque
import time
from utils import shuffle, get_gene_ontology


DATA_ROOT = 'data/fofe/'

go = get_gene_ontology()

go_prot = None


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

root_go_id = 'GO:0003674'
molecular_functions = get_subtree_set(root_go_id)


def get_anchestors(go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            q.append(parent_id)
    return go_set


def load_models():
    models = set()
    with open(DATA_ROOT + 'models.txt') as f:
        for line in f:
            line = line.strip()
            models |= get_anchestors(line)
    return models


def load_all_proteins():
    data = list()
    file_name = 'test.txt'
    models = load_models()
    with open(DATA_ROOT + file_name, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            go_set = set()
            for go_id in line[2].split('; '):
                if go_id in molecular_functions:
                    go_set |= get_anchestors(go_id).intersection(models)
            data.append((prot_id, go_set))
    return data


def load_pred_proteins():
    data = list()
    file_name = 'predictions-part.txt'
    with open(DATA_ROOT + file_name, 'r') as f:
        for line in f:
            line = line.strip().split()
            prot_id = line[0]
            go_set = set()
            for go_id in line[1:]:
                if go_id in go:
                    go_set |= get_anchestors(go_id)
            data.append((prot_id, go_set))
    return data


def get_proteins_by_go_id(data):
    res = dict()
    for prot_id, go_set in data:
        for go_id in go_set:
            if go_id not in res:
                res[go_id] = set()
            res[go_id].add(prot_id)
    return res


def main():
    start_time = time.time()
    print 'Loading data'
    real_data = load_all_proteins()
    print 'Loading predicted data'
    pred_data = load_pred_proteins()
    f = 0.0
    for i in range(len(pred_data)):
        prot1_id, gos = real_data[i]
        prot2_id, pred_gos = pred_data[i]
        if prot1_id == prot2_id:
            tp = len(gos.intersection(pred_gos))
            fp = len(pred_gos - gos)
            fn = len(gos - pred_gos)
            recall = tp / (1.0 * (tp + fn))
            precision = tp / (1.0 * (tp + fp))
            f += 2 * precision * recall / (precision + recall)

    print f / len(pred_data)
    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
