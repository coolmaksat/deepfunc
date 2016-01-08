#!/usr/bin/env python
import sys
from collections import deque
import time
from utils import shuffle, get_gene_ontology


DATA_ROOT = 'data/molecular_functions/'
RESULT_ROOT = 'data/molecular_functions/paac'
FILES = (
    'unseen-gos.txt',
    'predictions.txt')

go = get_gene_ontology()

go_prot = None


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
molecular_functions = get_go_set(root_go_id)


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


def load_all_proteins():
    data = list()
    file_name = FILES[0]
    with open(DATA_ROOT + file_name, 'r') as f:
        l = 0
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            go_set = set()
            for go_id in line[1].split('; '):
                if go_id in molecular_functions:
                    go_set.add(go_id)
            data.append((prot_id, go_set))
            l += 1
            if l > 60200:
                break
    return data


def load_pred_proteins():
    data = list()
    file_name = FILES[1]
    with open(DATA_ROOT + file_name, 'r') as f:
        l = 0
        for line in f:
            line = line.strip().split()
            prot_id = line[0]
            go_set = set()
            for go_id in line[1:]:
                if go_id in go:
                    go_set.add(go_id)
            data.append((prot_id, go_set))
            l += 1
            if l > 60200:
                break
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
    with open(DATA_ROOT + 'prediction-results.txt', 'w') as f:
        x = 0
        y = 0
        z = 0
        for i in range(len(pred_data)):
            go_set = set()
            prot1_id, gos = real_data[i]
            for go_id in gos:
                go_set |= get_go_set(go_id)
            prot2_id, pred_gos = pred_data[i]
            if prot1_id != prot2_id:
                sys.exit('Proteins %s %s differ on line %d' % (
                    i, prot1_id, prot2_id))
            correct = 0
            rcorrect = 0
            for go_id in pred_gos:
                if go_id in go_set:
                    correct += 1
                if go_id in gos:
                    rcorrect += 1
            if correct > 0:
                x += 1
            if correct > 0 and correct == len(pred_gos):
                y += 1
            if rcorrect > 0:
                z += 1
            f.write(
                prot1_id + ' ' + str(len(gos)) +
                ' ' + str(correct) + ' ' + str(rcorrect) +
                ' ' + str(len(pred_gos)))
            f.write('\n')
        print x, y, z

    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
