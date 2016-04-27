#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_fofe_hierarchical.py
'''

import numpy
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten, Highway)
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import (
    shuffle, train_val_test_split,
    get_gene_ontology, encode_seq_one_hot)
import sys
import os
from collections import deque
import pandas as pd
from multiprocessing import Pool

LAMBDA = 24
DATA_ROOT = 'data/fofe/'

go = get_gene_ontology()
go_model = dict()
classifier = None


def load_data(parent_id, go_id):
    pass


def get_model(
        go_id,
        parent_id,
        level):
    filepath = DATA_ROOT + 'level_' + str(level) + '/' + parent_id + '/' + go_id + '.hdf5'
    if not os.path.exists(filepath):
        return None
    key = parent_id + "_" + go_id
    if key in go_model:
        return go_model[key]
    model = Sequential()
    model.add(Dense(8000, activation='relu', input_dim=8000))
    model.add(Highway())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
    # Loading saved weights
    print 'Loading weights for ' + parent_id + '-' + go_id
    try:
        model.load_weights(filepath)
    except Exception, e:
        print 'Could not load weights for %s %s %d' % (parent_id, go_id, level)
        return None

    go_model[key] = model
    return model


def get_go_classifier(go_id, level):
    global model_n
    if level < 2:
        for ch_id in go[go_id]['children']:
            model = get_model(ch_id, go_id, level + 1)
            go[ch_id]['model'] = model
            if model is not None:
                get_go_classifier(ch_id, level + 1)
    return go[go_id]


def load_unseen_proteins():
    df = pd.read_pickle(DATA_ROOT + 'test.pkl')
    return df


def predict_functions(df):
    q = deque()
    q.append(classifier)
    functions = list()
    data = df[1]['data']
    data = data.reshape(1, data.shape[0])
    while len(q) > 0:
        x = q.popleft()
        ok = True
        for ch_id in x['children']:
            if 'model' in go[ch_id] and go[ch_id]['model']:
                model = go[ch_id]['model']
                pred = model.predict_classes(
                    data,
                    batch_size=1,
                    verbose=0)
                if pred[0][0] == 1:
                    ok = False
                    q.append(go[ch_id])
        if ok:
            functions.append(x['id'])
    return df[1]['proteins'], functions


def main(*args, **kwargs):
    global classifier
    classifier = get_go_classifier('GO:0003674', 0)
    print 'Total number of models %d' % (len(go_model), )
    print 'Loading unseen proteins'
    df = load_unseen_proteins()
    p = Pool(32)
    predictions = p.map(predict_functions, df.iterrows())
    with open(DATA_ROOT + 'predictions.txt', 'w') as f:
        for protein, functions in predictions:
            f.write(protein)
            for func in functions:
                f.write(' ' + func)
            f.write('\n')

if __name__ == '__main__':
    main(*sys.argv)
