#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_lstm.py
"""

import numpy
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Highway
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.layers.recurrent import LSTM, GRU
# from seya.layers.recurrent import Bidirectional

from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import train_test_split, normalize_aa
from sklearn.preprocessing import OneHotEncoder
from utils import (
    shuffle, encode_seq_one_hot)
import sys
import pdb

AALETTER = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


DATA_ROOT = 'data/recurrent/level_1/'
MAXLEN = 500

# forward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
# backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
# brnn = Bidirectional(
    # forward=forward_lstm, backward=backward_lstm, return_sequences=True)


def load_data(go_id):
    data = list()
    labels = list()
    pos = 1
    positive = list()
    negative = list()
    with open(DATA_ROOT + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq = []
            seq = encode_seq_one_hot(line[2][:500], maxlen=MAXLEN)

            if label == pos:
                positive.append(seq)
            else:
                negative.append(seq)
    shuffle(negative, seed=0)
    n = len(positive)
    data = negative[:n] + positive
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data, labels, seed=0)
    return numpy.array(labels), numpy.array(data, dtype="float32")


def model(labels, data, go_id):
    # set parameters:

    # Convolution
    filter_length = 7
    nb_filter = 64
    pool_length = 2
    k=7
    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 32
    nb_epoch = 12

    train, test = train_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    test_label, test_data = test
    test_label_rep = test_label

    model = Sequential()
    model.add(Convolution1D(
        input_dim=20,
        input_length=500,
        nb_filter=320,
        filter_length=20,
        border_mode="valid",
        activation="relu",
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=10, stride=10))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        nb_filter=320,
        filter_length=20,
        border_mode="valid",
        activation="relu",
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=10, stride=10))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Highway())
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1000))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    print 'compiling model'
    model.compile(
        loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
    print 'running at most 60 epochs'
    model_path = DATA_ROOT + go_id + '.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(
        train_data, train_label, batch_size=batch_size,
        nb_epoch=60, shuffle=True, show_accuracy=True,
        validation_split=0.3,
        callbacks=[checkpointer, earlystopper])

    # # Loading saved weights
    print 'Loading weights'
    model.load_weights(model_path)
    pred_data = model.predict_classes(test_data, batch_size=batch_size)
    # Saving the model
    # tresults = model.evaluate(test_data, test_label,show_accuracy=True)
    # print tresults
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
