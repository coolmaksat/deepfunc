#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gen_next_level_data.py
'''

import numpy
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten, Merge, Highway)
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import (
    train_test_split, normalize_aa,
    shuffle, encode_seq_one_hot, encode_seq, encode_seq_hydro,
    get_gene_ontology
)
import os
import sys
import pdb
from keras.optimizers import Adam
import shutil
from collections import deque
import pandas as pd

LAMBDA = 24
DATA_ROOT = 'data/fofe/'
CUR_LEVEL = 'level_1/'
NEXT_LEVEL = 'level_2/'

go = get_gene_ontology()
go_model = dict()

MAXLEN = 500


def get_gos_by_prot_id():
    data = dict()
    with open(DATA_ROOT + 'train.txt', 'r') as f:
        prot_id = 0
        for line in f:
            line = line.strip().split('\t')
            gos = line[2].split('; ')
            go_set = set()
            for go_id in gos:
                go_set.add(go_id)
            data[prot_id] = go_set
            prot_id += 1
    return data


def load_data(parent_id, go_id):
    df = pd.read_pickle(
        DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.pkl')
    n = 0
    for l in df['labels']:
        if l == 1:
            break
        n += 1
    df = df[n:]
    df = df.reindex()
    return df


def get_model(go_id, parent_id):
    filepath = DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.hdf5'
    model = Sequential()
    model.add(Dense(8000, activation='relu', input_dim=8000))
    model.add(Highway())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
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
    df = load_data(parent_id, go_id)
    go_sets = get_gos_by_prot_id()
    model = get_model(go_id, parent_id)
    data = df['data'].as_matrix()
    data = numpy.hstack(data).reshape(data.shape[0], 8000)
    pred = model.predict_classes(
        data,
        batch_size=16,
        verbose=1)
    gos = list()
    index = list()
    for i in range(len(data)):
        if pred[i] == 1:
            index.append(df.index[i])
            gos.append(list(go_sets[df['proteins'][df.index[i]]]))
    df = df.reindex(index)
    df['gos'] = pd.Series(gos, index=df.index)
    dirpath = DATA_ROOT + NEXT_LEVEL + 'data/'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    df.to_pickle(DATA_ROOT + NEXT_LEVEL + 'data/' + go_id + '.pkl')
    model_path = DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.hdf5'
    dst_model_path = DATA_ROOT + NEXT_LEVEL + 'data/' + go_id + '.hdf5'
    shutil.copyfile(model_path, dst_model_path)

if __name__ == '__main__':
    main(*sys.argv)
