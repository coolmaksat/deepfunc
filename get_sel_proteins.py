#!/usr/bin/env python
import sys
from collections import deque
import time
from utils import shuffle, get_gene_ontology


DATA_ROOT = 'data/swiss2/'
RESULT_ROOT = 'data/swiss2/level_1/'
FILES = (
    'train.txt',)

go = get_gene_ontology()

go_prot = None
go_seq = None
MIN_LEN = 24


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


INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def get_paac_by_prot_id():
    data = dict()
    with open(DATA_ROOT + 'uniprot-swiss-mol-func-paac.txt') as f:
        for line in f:
            line = line.strip().split()
            paac = line[1]
            for i in range(2, len(line)):
                paac += ' ' + line[i]
            data[line[0]] = paac

    return data

prot_paac = get_paac_by_prot_id()


def load_all_proteins():
    data = list()
    global go_seq
    go_seq = dict()
    max_len = 0
    sum_len = 0
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line[1]) < MIN_LEN:
                    continue
                if max_len < len(line[1]):
                    max_len = len(line[1])
                sum_len += len(line[1])
                prot_id = line[0]
                seq = line[1]
                go_seq[prot_id] = seq
                go_set = set()
                for go_id in line[2].split('; '):
                    if go_id in go:
                        go_set.add(go_id)
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


def select_proteins(go_id, parent_go_set):
    node = go[go_id]
    pos_go_set = get_subtree_set(go_id)
    neg_go_set = parent_go_set - pos_go_set
    positives = set()
    for g_id in pos_go_set:
        if g_id in go_prot:
            positives |= go_prot[g_id]
    negatives = set()
    for g_id in neg_go_set:
        if g_id in go_prot:
            negatives |= go_prot[g_id]
    negatives = negatives - positives
    positives = list(positives)
    negatives = list(negatives)
    shuffle(positives, seed=10)
    shuffle(negatives, seed=10)
    min_len = min(len(positives), len(negatives))
    with open(RESULT_ROOT + go_id + '.txt', 'w') as f:
        for prot_id in negatives[:min_len]:
            f.write('0 ' + prot_id + ' ' + prot_paac[prot_id] + '\n')
        for prot_id in positives[:min_len]:
            f.write('1 ' + prot_id + ' ' + prot_paac[prot_id] + '\n')
    print 'Finished selection for ' + go_id
    # for ch_id in node['children']:
    #     select_proteins(ch_id, pos_go_set)


def main():
    start_time = time.time()
    print 'Loading data'
    data = load_all_proteins()
    global go_prot
    print 'Getting proteins by go_id'
    go_prot = get_proteins_by_go_id(data)
    root_go_id = 'GO:0003674'
    root = go[root_go_id]
    root_go_set = get_subtree_set(root_go_id)
    print 'Starting protein selection'
    for ch_id in root['children']:
        print ch_id
        select_proteins(ch_id, root_go_set)

    # end_time = time.time() - start_time
    # print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
