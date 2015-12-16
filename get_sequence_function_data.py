#!/usr/bin/env python
# Filename: get_sequence_function_data.py
import sys

INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def main():

    positive = dict()
    with open('data/uniprot-go-0019825.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            seq = line[2]
            if prot_id not in positive:
                positive[prot_id] = seq

    negative = dict()
    with open('data/uniprot-go-0030246.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            seq = line[2]
            if prot_id not in negative:
                negative[prot_id] = seq
    data = list()
    labels = list()
    for gene_id, seq in positive.iteritems():
        if gene_id not in negative:
            data.append(seq)
            labels.append(1)
    pos_len = len(labels)
    for gene_id, seq in negative.iteritems():
        if gene_id not in positive:
            data.append(seq)
            labels.append(0)
        #     pos_len -= 1
        # if pos_len == 0:
        #     break
    min_len = sys.maxint
    with open('data/obtained/go_sequence.txt', 'w') as f:
        for i in range(len(labels)):
            if len(data[i]) > 24 and isOk(data[i]):
                if min_len > len(data[i]):
                    min_len = len(data[i])
                f.write(str(labels[i]) + ' ' + data[i] + '\n')
    print 'Minimum length:', min_len

if __name__ == '__main__':
    main()
