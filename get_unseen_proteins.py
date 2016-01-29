#!/usr/bin/env python
import sys
import os
from collections import deque
import time
from utils import shuffle, get_gene_ontology


DATA_ROOT = 'data/'
RESULT_ROOT = 'data/swiss2/'
FILES = (
    'uniprot-swiss-mol-func.txt',)


def load_all_proteins_paac():
    prots = dict()
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            for line in f:
                line = line.strip().split()
                prot_id = line[0]
                paac = line[1:]
                prots[prot_id] = paac
    return prots


INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])
MIN_LEN = 24


def is_ok(seq):
    if len(seq) < MIN_LEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def load_all_proteins():
    prots = list()
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                prot_id = line[0]
                seq = line[1]
                gos = line[2]
                if is_ok(seq):
                    prots.append((prot_id, seq, gos))
    return prots


def load_train_proteins():
    prot_set = set()
    for root, dirs, files in os.walk(DATA_ROOT + 'hierarchical/'):
        for filename in files:
            with open(root + filename, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    prot_set.add(line[1])
    return prot_set


def load_unseen_proteins():
    prot_set = set()
    with open(DATA_ROOT + 'unseen.txt', 'r') as f:
        for line in f:
            line = line.strip().split()
            prot_id = line[0]
            prot_set.add(prot_id)
    return prot_set


def main():
    start_time = time.time()
    print 'Loading all proteins'
    all_prots = load_all_proteins()
    shuffle(all_prots, seed=0)
    split = 0.8
    train_len = int(len(all_prots) * split)
    # print 'Loading train proteins'
    # train_set = load_train_proteins()
    # all_set = set(all_prots.keys())
    # print len(all_set), len(train_set)
    # unseen = all_set - train_set
    with open(RESULT_ROOT + 'train.txt', 'w') as f:
        for prot_id, seq, gos in all_prots[:train_len]:
            f.write(prot_id + '\t' + seq + '\t' + gos + '\n')
    with open(RESULT_ROOT + 'test.txt', 'w') as f:
        for prot_id, seq, gos in all_prots[train_len:]:
            f.write(prot_id + '\t' + seq + '\t' + gos + '\n')

    # print 'Loading unseen proteins'
    # unseen = load_unseen_proteins()
    # print 'Loading all proteins'
    # all_prots = load_all_proteins()
    # with open(DATA_ROOT + 'unseen-gos.txt', 'w') as f:
    #     for prot_id in unseen:
    #         f.write(prot_id)
    #         f.write('\t' + all_prots[prot_id] + '\n')

    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
