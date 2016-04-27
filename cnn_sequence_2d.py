#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_sequence_2d.py
'''

import numpy
from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling1D
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import (
    train_test_split, normalize_aa,
    shuffle, encode_seq_one_hot, encode_seq
)
import sys
import pdb
from keras.optimizers import Adam
from aaindex import (
    LUTR910101, NORM_AACOOC, AAINDEX,
    ALTS910101, CROG050101, OGAK980101,
    KOLA920101, TOBD000101, MEHP950102, RUSR970103)


MAXLEN = 500
DATA_ROOT = 'data/cnn/'
CUR_LEVEL = 'level_1/'
NEXT_LEVEL = 'level_2/'
DATA_ROOT += CUR_LEVEL


def load_data(parent_id, go_id):
    data1 = list()
    data2 = list()
    labels = list()
    positive1 = list()
    negative1 = list()
    positive2 = list()
    negative2 = list()

    with open(DATA_ROOT + parent_id + '/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq = line[2][:MAXLEN]
            sq1 = encode_seq_one_hot(seq, maxlen=MAXLEN)
            sq2 = encode_seq(OGAK980101, seq, maxlen=MAXLEN)
            sq3 = encode_seq(MEHP950102, seq, maxlen=MAXLEN)
            sq4 = encode_seq(CROG050101, seq, maxlen=MAXLEN)
            sq5 = encode_seq(TOBD000101, seq, maxlen=MAXLEN)
            sq6 = encode_seq(ALTS910101, seq, maxlen=MAXLEN)
            if label == 1:
                positive1.append([sq1])
                positive2.append(sq1)
            else:
                negative1.append([sq1])
                negative2.append(sq1)
    shuffle(negative1, negative2, seed=0)
    n = min(len(positive1), len(negative1))
    data1 = negative1[:n] + positive1[:n]
    data2 = negative2[:n] + positive2[:n]
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data1, data2, labels, seed=0)
    data = (
        numpy.array(data1, dtype='float32'),
        numpy.array(data2, dtype='float32'))
    return (numpy.array(labels), data)


def model(labels, data, parent_id, go_id):
    # set parameters:
    # Convolution
    nb_filter = 64
    nb_row = 5
    nb_col = 1

    pool_length = 3

    # Training
    batch_size = 64
    nb_epoch = 24

    lstm_size = 70

    data1, data2 = data

    train1, test1 = train_test_split(
        labels, data1, batch_size=batch_size, split=0.8)
    train_label, train1_data = train1

    train2, test2 = train_test_split(
        labels, data2, batch_size=batch_size, split=0.8)
    train_label, train2_data = train2

    if len(train1_data) < 100:
        raise Exception("No training data for " + go_id)

    test_label, test1_data = test1
    test_label, test2_data = test2
    test_label_rep = test_label

    first = Sequential()
    first.add(Convolution2D(
        nb_filter, nb_row, nb_col,
        border_mode='valid',
        input_shape=(1, MAXLEN, 20)))
    first.add(Activation('relu'))
    first.add(Convolution2D(2 * nb_filter, nb_row, nb_col))
    first.add(Activation('relu'))
    # first.add(Convolution2D(nb_filter, nb_row, nb_col))
    # first.add(Activation('relu'))
    first.add(MaxPooling2D(pool_size=(pool_length, 1)))
    first.add(Dropout(0.5))
    first.add(Flatten())

    second = Sequential()
    second.add(
        LSTM(lstm_size, return_sequences=True, input_shape=(MAXLEN, 20)))
    second.add(Dropout(0.25))
    # second.add(LSTM(lstm_size, return_sequences=True))
    # second.add(Dropout(0.25))
    second.add(LSTM(lstm_size, return_sequences=False))
    second.add(Dropout(0.25))
    second.add(Flatten())

    model = Sequential()
    model.add(Merge([first, second], mode='concat'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=0.00001)
    model.compile(
        loss='binary_crossentropy', optimizer=adam, class_mode='binary')

    model_path = DATA_ROOT + parent_id + '/' + go_id + '.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(
        X=[train1_data, train2_data], y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_split=0.3,
        callbacks=[checkpointer, earlystopper])

    model.load_weights(model_path)
    pred_data = model.predict_classes(
        [test1_data, test2_data],
        batch_size=batch_size)
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
        DATA_ROOT = 'data/cnn/' + CUR_LEVEL
    print 'Starting binary classification for ' + parent_id + '-' + go_id
    labels, data = load_data(parent_id, go_id)
    report = model(labels, data, parent_id, go_id)
    print report
    print_report(report, parent_id, go_id)

if __name__ == '__main__':
    main(*sys.argv)
