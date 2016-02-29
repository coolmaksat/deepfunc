#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gen_next_level_data.py
'''

import numpy
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import (
    shuffle, train_val_test_split,
    get_gene_ontology,
    encode_seq_one_hot)
from keras.optimizers import Adam

import sys
import os
from collections import deque

LAMBDA = 24
DATA_ROOT = 'data/cnn/'
CUR_LEVEL = 'level_1/'
NEXT_LEVEL = 'level_2/'

go = get_gene_ontology()
go_model = dict()

MAXLEN = 500


def get_gos_by_prot_id():
    data = dict()
    with open(DATA_ROOT + 'train.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            gos = line[2].split('; ')
            go_set = set()
            for go_id in gos:
                go_set.add(go_id)
            data[prot_id] = go_set
    return data


def load_data(parent_id, go_id):
    data = list()
    with open(DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == '1':
                prot_id = line[1]
                seq = line[2]
                data.append((prot_id, seq))
    return data


def get_model(
        go_id,
        parent_id,
        nb_filter=64,
        nb_row=3,
        nb_col=3,
        pool_length=2):
    filepath = DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.hdf5'
    model = Sequential()

    model.add(Convolution2D(nb_filter, nb_row, nb_col,
                            border_mode='valid',
                            input_shape=(1, MAXLEN, 20)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_length, pool_length)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = Adam(lr=0.00001)
    model.compile(
        loss='binary_crossentropy', optimizer=adam, class_mode='binary')
    # Loading saved weights
    print 'Loading weights for ' + go_id
    model.load_weights(filepath)
    return model


def main(*args, **kwargs):
    if len(args) < 3:
        raise Exception('Please provide function id')
    parent_id = args[1]
    go_id = args[2]
    if len(args) == 4:
        level = int(args[3])
        global CUR_LEVEL
        global NEXT_LEVEL
        CUR_LEVEL = 'level_' + str(level) + '/'
        NEXT_LEVEL = 'level_' + str(level + 1) + '/'
    data = load_data(parent_id, go_id)
    go_sets = get_gos_by_prot_id()
    model = get_model(go_id, parent_id)
    hots = list()
    for prot_id, seq in data:
        hots.append(encode_seq_one_hot(seq, maxlen=MAXLEN))
    hots = numpy.array(hots)
    hots = hots.reshape(hots.shape[0], 1, MAXLEN, 20)
    pred = model.predict_classes(
        hots,
        batch_size=16,
        verbose=1)
    result = list()
    for i in range(len(data)):
        if pred[i] == 1:
            result.append((data[i][0], data[i][1], list(go_sets[data[i][0]])))
    dirpath = DATA_ROOT + NEXT_LEVEL + 'data/'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(DATA_ROOT + NEXT_LEVEL + 'data/' + go_id + '.txt', 'w') as f:
        for prot_id, seq, go_list in result:
            f.write(prot_id + '\t' + seq + '\t' + go_list[0])
            for go_id in go_list[1:]:
                f.write(', ' + go_id)
            f.write('\n')


if __name__ == '__main__':
    main(*sys.argv)
