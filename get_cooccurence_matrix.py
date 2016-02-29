#!/usr/bin/env python
import sys
from collections import deque
import time
from utils import shuffle, AAINDEX, AANUM


DATA_ROOT = 'data/'
FILES = (
    'uniprot-swiss-mol-func.txt',)

INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])
MIN_LEN = 0
# C is the number of neighbor AA's to count as cooccurance
C = 2


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def load_all_sequences():
    data = list()
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                seq = line[1]
                if len(seq) > MIN_LEN and isOk(seq):
                    data.append(seq)
    return data


def main():
    seqs = load_all_sequences()
    mat = list()
    for i in range(AANUM):
        mat.append([0] * AANUM)
    for seq in seqs:
        for i in range(len(seq)):
            ind = AAINDEX[seq[i]]
            for j in range(i - C, i + C + 1):
                if i != j and j >= 0 and j < len(seq):
                    mat[ind][AAINDEX[seq[j]]] += 1
    print mat
if __name__ == '__main__':
    main()
