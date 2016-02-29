#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn_sequence.py
"""

import numpy
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import train_test_split, shuffle
import sys
from aaindex import (
        AAINDEX,
        CHAM810101,
        NORM_AAINDEX
    )


MAXLEN = 200
DATA_ROOT = 'data/recurrent/level_1/'


def load_data(go_id):
    data1 = list()
    data2 = list()
    labels = list()
    positive1 = list()
    negative1 = list()
    positive2 = list()
    negative2 = list()
    with open(DATA_ROOT + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq1 = []
            seq2 = []
            for x in line[2][:MAXLEN]:
                seq1.append(CHAM810101[AAINDEX[x]])
                seq2.append(NORM_AAINDEX[AAINDEX[x]])
            if label == 1:
                positive1.append(seq1)
                positive2.append(seq2)
            else:
                negative1.append(seq1)
                negative2.append(seq2)
    shuffle(negative1)
    shuffle(negative2)
    n = min(len(positive1), len(negative1))
    data1 = negative1[:n] + positive1[:n]
    data2 = negative2[:n] + positive2[:n]
    labels = [0] * n + [1] * n
    shuffle(data1, data2, labels)
    data1 = sequence.pad_sequences(data1, maxlen=MAXLEN, padding='post')
    data2 = sequence.pad_sequences(data2, maxlen=MAXLEN, padding='post')
    return (
        numpy.array(labels),
        numpy.array(data1, dtype="float32"),
        numpy.array(data2, dtype="float32"))


def model(labels, data1, data2, go_id):
    # set parameters:
    max_features = len(AAINDEX) + 1
    embedding_size = 128
    batch_size = 16
    nb_epoch = 10
    lstm_size = 128

    train1, test1 = train_test_split(
        labels, data1, batch_size=batch_size)
    train2, test2 = train_test_split(
        labels, data2, batch_size=batch_size)

    train_label, train1_data = train1
    train_label, train2_data = train2

    test_label, test1_data = test1
    test_label, test2_data = test2

    test_label_rep = test_label
    # 256 0.5 256
    first = Sequential()
    first.add(Embedding(
        max_features, embedding_size,
        input_length=MAXLEN))
    first.add(LSTM(lstm_size))

    second = Sequential()
    second.add(Embedding(
        max_features, embedding_size,
        input_length=MAXLEN))
    second.add(Dropout(0.5))
    second.add(LSTM(lstm_size))

    model = Sequential()
    model.add(Merge([first, second], mode='concat'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    model.fit(
        X=[train1_data, train2_data], y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_split=0.2)

    # model = Graph()
    # model.add_input(name='input', input_shape=(MAXLEN,), dtype=int)
    # model.add_node(Embedding(
    #     max_features, 128, input_length=MAXLEN, mask_zero=True),
    #                name='embedding', input='input')
    # model.add_node(LSTM(64), name='forward', input='embedding')
    # model.add_node(
    #     LSTM(64, go_backwards=True), name='backward', input='embedding')
    # model.add_node(
    #     Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
    # model.add_node(
    #     Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
    # model.add_output(name='output', input='sigmoid')

    # # try using different optimizers and different optimizer configs
    # model.compile('adam', {'output': 'binary_crossentropy'})
    # model.fit(
    #     {'input': train_data, 'output': train_label},
    #     batch_size=batch_size,
    #     nb_epoch=nb_epoch,
    #     validation_split=0.2)
    # pred_data = model.predict({'input': test_data}, batch_size=batch_size)
    # pred_data = numpy.round(numpy.array(pred_data['output']))
    # # Loading saved weights
    # print 'Loading weights'
    # model.load_weights(DATA_ROOT + go_id + '.hdf5')
    pred_data = model.predict_classes(
        [test1_data, test2_data],
        batch_size=batch_size)
    # Saving the model
    # print 'Saving the model for ' + go_id
    # model.save_weights(DATA_ROOT + go_id + '.hdf5', overwrite=True)
    return classification_report(list(test_label_rep), pred_data)


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    if len(args) != 2:
        sys.exit('Please provide GO Id')
    go_id = args[1]
    print 'Starting binary classification for ' + go_id
    labels, data1, data2 = load_data(go_id)
    report = model(labels, data1, data2, go_id)
    print report
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
