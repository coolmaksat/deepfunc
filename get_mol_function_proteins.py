#!/usr/bin/env python
# Filename: get_sequence_function_data.py
import sys
from collections import deque
import time


DATA_ROOT = 'data/'
FILES = (
    'uniprot-swiss.txt',)
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])


def get_gene_ontology():
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open('data/go.obo', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
    if obj is not None:
        go[obj['id']] = obj
    for go_id, val in go.iteritems():
        if 'children' not in val:
            val['children'] = list()
        for g_id in val['is_a']:
            if 'children' not in go[g_id]:
                go[g_id]['children'] = list()
            go[g_id]['children'].append(go_id)
    return go


def get_go_set(go_id):
    go = get_gene_ontology()
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def main():
    start_time = time.time()
    go_set = get_go_set('GO:0003674')
    print 'Starting filtering proteins with set of %d GOs' % (len(go_set),)
    min_len = sys.maxint
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            with open(DATA_ROOT + 'uniprot-swiss-mol-func.txt', 'w') as fall:
                for line in f:
                    line = line.strip().split('\t')
                    prot_id = line[0]
                    seq = line[1]
                    seq_len = len(seq)
                    gos = line[2]
                    if seq_len > 24 and isOk(seq):
                        go_ok = False
                        for go in gos.split('; '):
                            if go in go_set:
                                go_ok = True
                                break
                        if go_ok:
                            fall.write(
                                prot_id + '\t' + seq + '\t' + gos + '\n')
                        if min_len > seq_len:
                            min_len = seq_len
    end_time = time.time() - start_time
    print 'Minimum length of sequences:', min_len
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
