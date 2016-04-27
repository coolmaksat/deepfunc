#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_sequence.py
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
    shuffle, encode_seq_one_hot, encode_seq
)
import os
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
nb_classes = 0


def load_data(parent_id, go_id):
    data = list()
    labels = list()
    global nb_classes
    with open(DATA_ROOT + parent_id + '/' + go_id + '.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            seq = line[1][:MAXLEN]
            labs = line[2].split('|')
            data.append(seq)
            for i in range(len(labs)):
                labs[i] = int(labs[i])
                nb_classes = max(nb_classes, labs[i])
            labels.append(labs)
    nb_classes += 1
    for i in range(len(labels)):
        l = [0] * nb_classes
        for x in labels[i]:
            l[x] = 1
        labels[i] = l
    for i in range(len(data)):
        data[i] = encode_seq_one_hot(data[i], maxlen=MAXLEN)
    shuffle(data, labels, seed=0)
    return numpy.array(
        labels, dtype='float32'), numpy.array(data, dtype='float32')


def model(labels, data, parent_id, go_id):
    # Convolution
    filter_length = 20
    nb_filter = 32
    pool_length = 10
    global nb_classes

    # LSTM
    lstm_output_size = 128

    # Training
    batch_size = 64
    nb_epoch = 64

    train, test = train_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train
    # sample_weight = [1.0 if y == 1 else 1.0 for y in train_label]
    # sample_wseight = numpy.array(sample_weight, dtype='float32')

    test_label, test_data = test
    test_label_rep = test_label

    model = Sequential()
    model.add(Convolution1D(input_dim=20,
                            input_length=MAXLEN,
                            nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length, stride=10))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length, stride=10))
    model.add(LSTM(lstm_output_size))
    # model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop')

    model_path = DATA_ROOT + parent_id + '/' + go_id + '.hdf5'
    # parent_model_path = DATA_ROOT + 'data/' + parent_id + '.hdf5'
    # if os.path.exists(parent_model_path):
    #     print 'Loading parent model weights'
    #     model.load_weights(parent_model_path)
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopper])

    # Loading saved weights
    print 'Loading weights'
    model.load_weights(DATA_ROOT + parent_id + '/' + go_id + '.hdf5')
    score = model.evaluate(
        test_data, test_label, show_accuracy=True, verbose=1)
    print 'Score: ', score[0]
    print 'Accuracy: ', score[1]
    # pred_data = model.predict_classes(
    #     test_data, batch_size=batch_size)

    # return classification_report(list(test_label_rep), pred_data)


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
