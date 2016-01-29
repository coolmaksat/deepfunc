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
DATA_ROOT = 'data/swiss2/'

go = get_gene_ontology()
go_model = dict()


def load_data(go_id):
    data = list()
    with open(DATA_ROOT + 'level_1/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == '1':
                prot_id = line[1]
                paac = list()
                for v in line[2:]:
                    paac.append(float(v))
                data.append((prot_id, paac))
    return data


def get_model(
        go_id,
        max_features=5000,
        embedding_dims=100,
        nb_filters=250,
        hidden_dims=250,
        pool_length=2,
        filter_length=3):
    filepath = DATA_ROOT + 'level_1/models/' + go_id + '.hdf5'
    size = os.path.getsize(filepath)
    max_features = get_model_max_features(size)
    global go_model
    if go_id in go_model:
        return go_model[go_id]
    # length of APAAC
    maxlen = 20 + 6 * LAMBDA

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))
    model.add(Dropout(0.25))
    model.add(Convolution1D(
        input_dim=embedding_dims,
        nb_filter=nb_filters,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Flatten())
    output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2
    model.add(Dense(output_size, hidden_dims))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims, 1))
    model.add(Activation('sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    # Loading saved weights
    print 'Loading weights for ' + go_id
    model.load_weights(filepath)
    return model


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
