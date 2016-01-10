#!/usr/bin/env python

import numpy
from utils import (
    shuffle, train_val_test_split, get_gene_ontology, get_model_max_features)
import sys
import os

LAMBDA = 24
DATA_ROOT = 'data/swiss/'

go = get_gene_ontology()
go_model = dict()


def load_data(go_id):
    data = list()
    with open(DATA_ROOT + 'level_2/data/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split()
            prot_id = line[0]
            paac = list()
            for v in line[1:]:
                paac.append(float(v))
            data.append((prot_id, paac))
    return data


def main(*args, **kwargs):
    if len(args) != 2:
        raise Exception('Please provide function id')
    go_id = args[1]
    data = load_data(go_id)
    model = get_model(go_id)
    paacs = list()
    for prot_id, paac in data:
        paacs.append(paac)
    pred = model.predict_classes(
        numpy.array(paacs, dtype="float32"),
        batch_size=1,
        verbose=1)
    result = list()
    for i in range(len(data)):
        if pred[i] == 1:
            result.append(data[i])

    with open(DATA_ROOT + 'level_2/data/' + go_id + '.txt', 'w') as f:
        for prot_id, paac in result:
            f.write(prot_id)
            for p in paac:
                f.write(' ' + str(p))
            f.write('\n')


if __name__ == '__main__':
    main(*sys.argv)
