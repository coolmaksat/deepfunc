#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_sequence_hierarchical.py
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
    shuffle, train_val_test_split, get_gene_ontology)
import sys
import os
from collections import deque

LAMBDA = 24
DATA_ROOT = 'data/molecular_functions/paac/'

go = get_gene_ontology()
go_model = dict()


def load_data(go_id):
    pass


def get_model(
        go_id,
        max_features=10000,
        embedding_dims=100,
        nb_filters=250,
        hidden_dims=250,
        pool_length=2,
        filter_length=3):
    is_ok = False
    filepath = None
    if os.path.exists(DATA_ROOT + 'done/' + go_id + '.hdf5'):
        is_ok = True
        filepath = DATA_ROOT + 'done/' + go_id + '.hdf5'
    elif os.path.exists(DATA_ROOT + 'done-20000/' + go_id + '.hdf5'):
        is_ok = True
        max_features = 20000
        filepath = DATA_ROOT + 'done-20000/' + go_id + '.hdf5'
    elif os.path.exists(DATA_ROOT + 'done-40000/' + go_id + '.hdf5'):
        is_ok = True
        max_features = 40000
        filepath = DATA_ROOT + 'done-40000/' + go_id + '.hdf5'
    elif os.path.exists(DATA_ROOT + 'done-80000/' + go_id + '.hdf5'):
        is_ok = True
        max_features = 80000
        filepath = DATA_ROOT + 'done-80000/' + go_id + '.hdf5'
    if not is_ok:
        return None
    global go_model
    if go_id in go_model:
        return go_model[go_id]
    # length of APAAC
    maxlen = 20 + 2 * LAMBDA

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
    go_model[go_id] = model
    return model


def get_go_classifier(go_id, level):
    global model_n
    if level < 5:
        for ch_id in go[go_id]['children']:
            model = get_model(ch_id)
            go[ch_id]['model'] = model
            if model is not None:
                get_go_classifier(ch_id, level + 1)
    return go[go_id]


def load_unseen_proteins():
    prots = dict()
    with open(DATA_ROOT + 'unseen.txt', 'r') as f:
        for line in f:
            line = line.strip().split()
            prot_id = line[0]
            paac = list()
            for p in line[1:]:
                paac.append(float(p))
            if len(paac) == 20 + 2 * LAMBDA:
                prots[prot_id] = paac
    return prots


def predict_functions(classifier, paac):
    q = deque()
    q.append(classifier)
    functions = list()
    while len(q) > 0:
        x = q.popleft()
        ok = True
        for ch_id in x['children']:
            if 'model' in go[ch_id] and go[ch_id]['model']:
                model = go[ch_id]['model']
                pred = model.predict_classes(
                    numpy.array([paac], dtype="float32"),
                    batch_size=1,
                    verbose=0)
                if pred[0][0] == 0:
                    ok = False
                    q.append(go[ch_id])
        if ok:
            functions.append(x['id'])

    return functions


def main(*args, **kwargs):
    try:
        classifier = get_go_classifier('GO:0003674', 0)
        print 'Total number of models %d' % (len(go_model), )
        print 'Loading unseen proteins'
        prots = load_unseen_proteins()
        with open(DATA_ROOT + 'predictions.txt', 'w') as f:
            for prot_id, paac in prots.iteritems():
                f.write(prot_id)
                print 'Predicting functions for protein ' + prot_id
                functions = predict_functions(classifier, paac)
                print functions
                for func in functions:
                    f.write(' ' + func)
                f.write('\n')
    except Exception, e:
        print e

if __name__ == '__main__':
    main(*sys.argv)
