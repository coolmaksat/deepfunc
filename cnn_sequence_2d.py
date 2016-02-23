#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_sequence_2d.py


The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
The hydrophilicity values are from PNAS, 1981, 78:3824-3828
(T.P.Hopp & K.R.Woods). The side-chain mass for each of the 20 amino acids. CRC
Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton,
Florida (1985). R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones,
Data for Biochemical Research 3rd ed.,
Clarendon Press Oxford (1986).

'''

import numpy
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling1D
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
from utils import train_val_test_split, normalize_aa
import sys
import pdb
from keras.optimizers import Adam

AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

MAXLEN = 1000
DATA_ROOT = 'data/cnn/'
CUR_LEVEL = 'level_1/'
NEXT_LEVEL = 'level_2/'
DATA_ROOT += CUR_LEVEL

encoder = OneHotEncoder()


def init_encoder():
    data = list()
    for l in AALETTER:
        data.append([ord(l)])
    encoder.fit(data)

init_encoder()


def encode_seq(seq):
    data = list()
    for l in seq:
        data.append([ord(l)])
    data = encoder.transform(data).toarray()
    return list(data)


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


def load_data(parent_id, go_id):
    data = list()
    labels = list()
    pos = 1
    positive = list()
    negative = list()
    with open(DATA_ROOT + parent_id + '/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            seq = line[2][:MAXLEN]
            seq = encode_seq(seq)
            while len(seq) < MAXLEN:
                seq.append([0.0] * 20)
            if label == pos:
                positive.append(seq)
            else:
                negative.append(seq)
    shuffle(negative, seed=10)
    n = len(positive)
    data = negative[:n] + positive
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data, labels, seed=30)
    return numpy.array(labels), numpy.array(data, dtype='float32')


def model(labels, data, parent_id, go_id):
    # set parameters:
    # Convolution
    nb_filter = 64
    nb_row = 3
    nb_col = 1

    pool_length = 2

    # Training
    batch_size = 2
    if len(data) >= 512:
        batch_size = 32
    nb_epoch = 12

    train, val, test = train_val_test_split(
        labels, data, batch_size=batch_size, split=0.6)
    train_label, train_data = train

    if len(train_data) == 0:
        raise Exception("No training data for " + go_id)
    val_label, val_data = val
    test_label, test_data = test
    test_label_rep = test_label

    train_data = train_data.reshape(train_data.shape[0], 1, MAXLEN, 20)
    test_data = test_data.reshape(test_data.shape[0], 1, MAXLEN, 20)
    val_data = val_data.reshape(val_data.shape[0], 1, MAXLEN, 20)
    model = Sequential()

    model.add(Convolution2D(nb_filter, nb_row, nb_col,
                            border_mode='valid',
                            input_shape=(1, MAXLEN, 20)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter, nb_row, nb_col))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_length, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = Adam(lr=0.00001)
    model.compile(
        loss='binary_crossentropy', optimizer='adadelta', class_mode='binary')

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
    # model.save_weights(DATA_ROOT + parent_id + '/' + go_id + '.hdf5', overwrite=True)
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
