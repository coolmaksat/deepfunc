#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_fofe.py parent_id go_id
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
    shuffle, encode_seq_one_hot, encode_seq, encode_seq_hydro
)
import os
import sys
import pdb
from keras.optimizers import Adam
from aaindex import (
    LUTR910101, NORM_AACOOC, AAINDEX,
    ALTS910101, CROG050101, OGAK980101,
    KOLA920101, TOBD000101, MEHP950102, RUSR970103)
import pandas as pd


MAXLEN = 500
DATA_ROOT = 'data/fofe/'
CUR_LEVEL = 'level_1/'
NEXT_LEVEL = 'level_2/'
DATA_ROOT += CUR_LEVEL


def load_data(parent_id, go_id):
    df = pd.read_pickle(DATA_ROOT + parent_id + '/' + go_id + '.pkl')
    # shuffle(df, seed=0)
    # numpy.random.seed(0)
    df = df.reindex(numpy.random.permutation(df.index))
    return df


def model(df, parent_id, go_id):

    # Training
    batch_size = 64
    nb_epoch = 64

    # Split pandas DataFrame
    n = len(df)
    split = 0.8
    m = int(n * split)
    train, test = df[:m], df[m:]


    # train, test = train_test_split(
    #     labels, data, batch_size=batch_size)

    train_label, train_data = train['labels'], train['data']

    if len(train_data) < 100:
        raise Exception("No training data for " + go_id)

    test_label, test_data = test['labels'], test['data']
    test_label_rep = test_label


    train_data = train_data.as_matrix()

    test_data = test_data.as_matrix()
    train_data = numpy.hstack(train_data).reshape(train_data.shape[0], 8000)
    test_data = numpy.hstack(test_data).reshape(test_data.shape[0], 8000)
    shape = numpy.shape(train_data)

    print('X_train shape: ', shape)
    print('X_test shape: ', test_data.shape)
    model = Sequential()
    model.add(Dense(8000, activation='relu', input_dim=8000))
    model.add(Highway())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

    model_path = DATA_ROOT + parent_id + '/' + go_id + '.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopper])

    # Loading saved weights
    print 'Loading weights'
    model.load_weights(model_path)
    pred_data = model.predict_classes(
        test_data, batch_size=batch_size)
    return classification_report(list(test_label_rep), pred_data)


def print_report(report, parent_id, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + parent_id + '/' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    if len(args) < 3:
        sys.exit('Please provide parent id and GO Id')
    parent_id = args[1]
    go_id = args[2]
    if len(args) == 4:
        level = int(args[3])
        global CUR_LEVEL
        global NEXT_LEVEL
        global DATA_ROOT
        CUR_LEVEL = 'level_' + str(level) + '/'
        NEXT_LEVEL = 'level_' + str(level + 1) + '/'
        DATA_ROOT = 'data/fofe/' + CUR_LEVEL
    print 'Starting binary classification for ' + parent_id + '-' + go_id
    df = load_data(parent_id, go_id)
    report = model(df, parent_id, go_id)
    print report
    print_report(report, parent_id, go_id)


if __name__ == '__main__':
    main(*sys.argv)
