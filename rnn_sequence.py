#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn_sequence.py
"""

import numpy
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, Highway
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import (
    train_test_split, shuffle,
    encode_seq_one_hot, encode_seq, encode_seq_hydro)
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
from aaindex import (
    CHAM810101,
    NORM_AAINDEX,
    LUTR910101, NORM_AACOOC, AAINDEX,
    ALTS910101, CROG050101, OGAK980101,
    KOLA920101, TOBD000101, MEHP950102, RUSR970103)

MAXLEN = 500
DATA_ROOT = 'data/recurrent/level_1/'


def load_data(go_id):
    positive1 = list()
    positive2 = list()
    negative1 = list()
    negative2 = list()
    with open(DATA_ROOT + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq = line[2][:MAXLEN]
            hydro = encode_seq_hydro(seq, maxlen=MAXLEN)
            seq = encode_seq_one_hot(seq, maxlen=MAXLEN)
            if label == 1:
                positive1.append(seq)
                positive2.append(hydro)
            else:
                negative1.append(seq)
                negative2.append(hydro)
    shuffle(negative1, negative2, seed=0)
    n = len(positive1)
    data1 = negative1[:n] + positive1
    data2 = negative2[:n] + positive2
    labels = [0] * len(negative1) + [1] * len(positive1)
    shuffle(data1, data2, labels, seed=0)
    data = (
        numpy.array(data1, dtype='float32'),
        numpy.array(data2, dtype='float32'))
    return (
        numpy.array(labels, dtype='float32'),
        data)


def model(labels, data, go_id):
    # set parameters:
    batch_size = 64
    nb_epoch = 10
    lstm_size = 128

    data1, data2 = data

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
    model = Graph()
    model.add_input(name='input1', batch_input_shape=(batch_size, 20))
    model.add_input(name='input2', batch_input_shape=(batch_size, 3))
    model.add_node(Convolution1D(
        nb_filter=32,
        filter_length=20,
        border_mode='valid',
        activation='relu',
        subsample_length=1), name='conv1', input='input1')
    model.add_node(MaxPooling1D(
        pool_length=10, stride=10), name='pool1', input='conv1')
    model.add_node(
        LSTM(lstm_size), name='lstm1', input='pool1')
    model.add_node(Convolution1D(
        nb_filter=32,
        filter_length=3,
        border_mode='valid',
        activation='relu',
        subsample_length=1), name='conv2', input='input2')
    model.add_node(MaxPooling1D(
        pool_length=2), name='pool2', input='conv2')
    model.add_node(
        LSTM(lstm_size), name='lstm2', input='pool2')
    model.add_node(
        Dense(1024),
        name='dense1', inputs=['lstm1', 'lstm2'])
    model.add_node(Dropout(0.25), name='dropout', input='dense1')
    model.add_node(Activation('relu'), name='relu', input='dropout')
    model.add_node(
        Dense(1, activation='sigmoid'), name='dense2', input='relu')
    model.add_output(name='output', input='dense2')

    # try using different optimizers and different optimizer configs
    model.compile('adadelta', {'output': 'binary_crossentropy'})
    model_path = DATA_ROOT + go_id + '.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(
        {'input1': train1_data, 'input2': train2_data, 'output': train_label},
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopper])

    print 'Loading weights'
    model.load_weights(model_path)

    pred_data = model.predict(
        {'input1': test1_data, 'input2': test2_data}, batch_size=batch_size)
    pred_data = numpy.round(numpy.array(pred_data['output']))
    # Loading saved weights
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
    labels, data = load_data(go_id)
    report = model(labels, data, go_id)
    print report
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
