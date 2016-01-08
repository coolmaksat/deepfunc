#!/usr/bin/env python

import numpy
from utils import (
    shuffle, train_val_test_split, get_gene_ontology)
import sys
import os
from collections import deque

DATA_ROOT = 'data/molecular_functions/hierarchical/level_1/paac/'


def load_data(go_id):
    positives = list()
    negatives = list()
    with open(DATA_ROOT + go_id + '.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] == '0':
                negatives.append(line)
            else:
                positives.append(line)
    return (positives, negatives)


def main(*args, **kwargs):
    try:
        if len(args) != 3:
            raise Exception("Please provide go_id and number of proteins")
        go_id = args[1]
        positives, negatives = load_data(go_id)
        n = int(args[2])
        shuffle(positives)
        shuffle(negatives)
        with open(DATA_ROOT + go_id + '.small.txt', 'w') as f:
            for line in negatives[:n]:
                f.write(line + '\n')
            for line in positives[:n]:
                f.write(line + '\n')
    except Exception, e:
        print e

if __name__ == '__main__':
    main(*sys.argv)
