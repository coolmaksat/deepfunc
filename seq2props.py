#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gen_next_level_data.py
'''

import numpy
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten)
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import (
    shuffle, train_val_test_split, get_gene_ontology, get_model_max_features)
import sys
import os
from collections import deque

LAMBDA = 24
DATA_ROOT = 'data/recurrent/'

go = get_gene_ontology()
go_model = dict()

HYDROPHOBICITY = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08}

HYDROPHILICITY = {
    "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0,
    "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5,
    "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5}

RESIDUEMASS = {
    "A": 15.0, "R": 101.0, "N": 58.0, "D": 59.0, "C": 47.0, "Q": 72.0,
    "E": 73.0, "G": 1.000, "H": 82.0, "I": 57.0, "L": 57.0, "K": 73.0,
    "M": 75.0, "F": 91.0, "P": 42.0, "S": 31.0, "T": 45.0, "W": 130.0,
    "Y": 107.0, "V": 43.0}

PK1 = {
    "A": 2.35, "C": 1.71, "D": 1.88, "E": 2.19, "F": 2.58, "G": 2.34,
    "H": 1.78, "I": 2.32, "K": 2.20, "L": 2.36, "M": 2.28, "N": 2.18,
    "P": 1.99, "Q": 2.17, "R": 2.18, "S": 2.21, "T": 2.15, "V": 2.29,
    "W": 2.38, "Y": 2.20}

PK2 = {
    "A": 9.87, "C": 10.78, "D": 9.60, "E": 9.67, "F": 9.24, "G": 9.60,
    "H": 8.97, "I": 9.76, "K": 8.90, "L": 9.60, "M": 9.21, "N": 9.09,
    "P": 10.6, "Q": 9.13, "R": 9.09, "S": 9.15, "T": 9.12, "V": 9.74,
    "W": 9.39, "Y": 9.11}

PI = {
    "A": 6.11, "C": 5.02, "D": 2.98, "E": 3.08, "F": 5.91, "G": 6.06,
    "H": 7.64, "I": 6.04, "K": 9.47, "L": 6.04, "M": 5.74, "N": 10.76,
    "P": 6.30, "Q": 5.65, "R": 10.76, "S": 5.68, "T": 5.60, "V": 6.02,
    "W": 5.88, "Y": 5.63}


def load_data(go_id):
    data = list()
    with open(DATA_ROOT + 'level_1/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split()
            label = line[0]
            prot_id = line[1]
            seq = line[2]
            data.append((label, prot_id, seq))
    return data


def main(*args, **kwargs):
    if len(args) != 2:
        raise Exception('Please provide function id')
    go_id = args[1]
    data = load_data(go_id)
    props = [
        HYDROPHOBICITY,
        HYDROPHILICITY,
        RESIDUEMASS,
        PK1,
        PK2,
        PI
    ]
    with open(DATA_ROOT + 'level_1_props/' + go_id + '.txt', 'w') as f:
        for label, prot_id, seq in data:
            n = len(seq)
            for prop in props:
                f.write(label)
                for x in seq[:500]:
                    f.write('\t' + str(prop[x]))
                for x in range(n, 500):
                    f.write(' 0.0')
                f.write('\n')

if __name__ == '__main__':
    main(*sys.argv)
