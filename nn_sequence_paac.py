#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_sequence_paac.py
'''

import numpy
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten)
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import train_val_test_split
import sys

LAMBDA = 24
DATA_ROOT = 'data/swiss2/level_1/'


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
            paac = list()
            for i in range(2, len(line)):
                paac.append(float(line[i]))
            if len(paac) != 20 + 6 * LAMBDA:
                print 'Bad data in line %d' % ln
                continue
            if label == pos:
                positive.append(paac)
            else:
                negative.append(paac)
            ln += 1
    shuffle(negative, seed=10)
    n = len(positive)
    data = negative[:n] + positive
    labels = [0.0] * n + [1.0] * n
    # Previous was 30
    shuffle(data, labels, seed=30)
    return numpy.array(labels), numpy.array(data, dtype="float32")


def model(labels, data, go_id):
    # set parameters:
    max_features = 60000
    batch_size = 256
    embedding_dims = 100
    nb_filters = 250
    hidden_dims = 250
    nb_epoch = 12

    # pool lengths
    pool_length = 2
    # level of convolution to perform
    filter_length = 3

    # length of APAAC
    maxlen = 20 + 6 * LAMBDA

    train, val, test = train_val_test_split(
        labels, data, batch_size=batch_size)
    train_label, train_data = train

    val_label, val_data = val
    test_label, test_data = test

    test_label_rep = test_label

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))
    model.add(Dropout(0.25))
    model.add(Convolution1D(
        input_dim=embedding_dims,
        nb_filter=nb_filters,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Flatten())
    output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2
    model.add(Dense(output_size, hidden_dims))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims, 1))
    model.add(Activation('sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    weights_train = [1.0 if y == 1 else 1.0 for y in train_label]
    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_data=(val_data, val_label))
    # # Loading saved weights
    # print 'Loading weights'
    # model.load_weights(DATA_ROOT + go_id + '.hdf5')
    score = model.evaluate(
        test_data, test_label,
        batch_size=batch_size, verbose=1, show_accuracy=True)
    print "Loss:", score[0], "Accuracy:", score[1]
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
    print_report(report, go_id)

if __name__ == '__main__':
    main(*sys.argv)
