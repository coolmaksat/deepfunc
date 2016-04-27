#!/usr/bin/env python
import sys
from collections import deque
import time
from utils import shuffle, get_gene_ontology


DATA_ROOT = 'data/cnn/'
RESULT_ROOT = 'data/cnn/level_1/GO:0003674/'
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

# prot_paac = get_paac_by_prot_id()


def load_all_proteins():
    data = list()
    global go_seq
    go_seq = dict()
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line[1]) < MIN_LEN:
                    continue
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


def select_proteins(go_id):
    node = go[go_id]

    with open(RESULT_ROOT + go_id + '.txt', 'w') as f:
        children = list()
        prot_set = set()
        ch_prots = dict()
        for i in range(len(node['children'])):
            ch_id = node['children'][i]
            ch_prots[ch_id] = set()
            ch_go_set = get_subtree_set(ch_id)
            for g_id in ch_go_set:
                if g_id in go_prot:
                    ch_prots[ch_id] |= go_prot[g_id]
            if len(ch_prots[ch_id]) > 199:
                children.append(ch_id)
                prot_set |= ch_prots[ch_id]
        ch_ind = dict()
        for i in range(len(children)):
            ch_ind[children[i]] = i
        for prot_id in prot_set:
            f.write(prot_id + '\t' + go_seq[prot_id] + '\t')
            labels = list()
            for g_id, prots in ch_prots.iteritems():
                if prot_id in prots:
                    if g_id in ch_ind:
                        labels.append(ch_ind[g_id])
            if labels:
                f.write(str(labels[0]))
                for l in labels[1:]:
                    f.write('|' + str(l))
                f.write('\n')


def main():
    start_time = time.time()
    print 'Loading data'
    data = load_all_proteins()
    global go_prot
    print 'Getting proteins by go_id'
    go_prot = get_proteins_by_go_id(data)
    root_go_id = 'GO:0003674'
    print 'Starting protein selection'
    select_proteins(root_go_id)
    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
