#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn_sequence.py


The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
The hydrophilicity values are from PNAS, 1981, 78:3824-3828
(T.P.Hopp & K.R.Woods). The side-chain mass for each of the 20 amino acids. CRC
Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton,
Florida (1985). R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones,
Data for Biochemical Research 3rd ed.,
Clarendon Press Oxford (1986).

"""

import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import train_val_test_split, normalize_aa
import sys


AALETTER = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

HYDROPHOBICITY = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08}

HYDROPHILICITY = {
    "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0,
    "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5,
    "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5}

RESIDUEMASS = {
    "A": 15.0, "R": 101.0, "N": 58.0, "D": 59.0, "C": 47.0, "Q": 72.0,
    "E": 73.0, "G": 1.000, "H": 82.0, "I": 57.0, "L": 57.0, "K": 73.0,
    "M": 75.0, "F": 91.0, "P": 42.0, "S": 31.0, "T": 45.0, "W": 130.0,
    "Y": 107.0, "V": 43.0}

PK1 = {
    "A": 2.35, "C": 1.71, "D": 1.88, "E": 2.19, "F": 2.58, "G": 2.34,
    "H": 1.78, "I": 2.32, "K": 2.20, "L": 2.36, "M": 2.28, "N": 2.18,
    "P": 1.99, "Q": 2.17, "R": 2.18, "S": 2.21, "T": 2.15, "V": 2.29,
    "W": 2.38, "Y": 2.20}

PK2 = {
    "A": 9.87, "C": 10.78, "D": 9.60, "E": 9.67, "F": 9.24, "G": 9.60,
    "H": 8.97, "I": 9.76, "K": 8.90, "L": 9.60, "M": 9.21, "N": 9.09,
    "P": 10.6, "Q": 9.13, "R": 9.09, "S": 9.15, "T": 9.12, "V": 9.74,
    "W": 9.39, "Y": 9.11}

PI = {
    "A": 6.11, "C": 5.02, "D": 2.98, "E": 3.08, "F": 5.91, "G": 6.06,
    "H": 7.64, "I": 6.04, "K": 9.47, "L": 6.04, "M": 5.74, "N": 10.76,
    "P": 6.30, "Q": 5.65, "R": 10.76, "S": 5.68, "T": 5.60, "V": 6.02,
    "W": 5.88, "Y": 5.63}

HYDROPHILICITY = normalize_aa(HYDROPHILICITY)
HYDROPHOBICITY = normalize_aa(HYDROPHOBICITY)
RESIDUEMASS = normalize_aa(RESIDUEMASS)

LAMBDA = 24
DATA_ROOT = 'data/recurrent/level_1/'


def shuffle(*args, **kwargs):
    seed = None
    if 'seed' in kwargs:
        seed = kwargs['seed']
    rng_state = numpy.random.get_state()
    for arg in args:
        if seed is not None:
            numpy.random.seed(seed)
        else:
            numpy.random.set_state(rng_state)
        numpy.random.shuffle(arg)


def load_data(go_id):
    data = list()
    labels = list()
    pos = 1
    positive = list()
    negative = list()
    ln = 0
    with open(DATA_ROOT + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq = []
            for x in line[2]:
                seq.append(ord(x) - ord('A'))
            if label == pos:
                positive.append(seq)
            else:
                negative.append(seq)
            ln += 1
    shuffle(negative, seed=10)
    n = len(positive)
    data = negative[:n] + positive
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data, labels, seed=30)
    maxlen = 500
    data = sequence.pad_sequences(data, maxlen=maxlen)
    return numpy.array(labels), numpy.array(data, dtype="float32")


def model(labels, data, go_id):
    # set parameters:
    max_features = 5000
    batch_size = 16
    nb_epoch = 12
    maxlen = 500
    train, val, test = train_val_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    val_label, val_data = val
    test_label, test_data = test

    test_label_rep = test_label

    model = Sequential()
    model.add(Embedding(
        max_features, 8, mask_zero=True))
    model.add(LSTM(8, 8))
    model.add(Dropout(0.5))
    model.add(Dense(8, 1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')

    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_data=(val_data, val_label))
    # # Loading saved weights
    # print 'Loading weights'
    # model.load_weights(DATA_ROOT + go_id + '.hdf5')
    pred_data = model.predict_classes(test_data, batch_size=batch_size)
    # Saving the model
    print 'Saving the model for ' + go_id
    model.save_weights(DATA_ROOT + go_id + '.hdf5', overwrite=True)
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
    # print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
