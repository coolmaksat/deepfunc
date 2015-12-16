#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_sequence.py
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

LAMBDA = 24


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


def load_data():
    data = list()
    labels = list()
    with open('data/obtained/go_multi_paac_l.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            label = int(line[0])
            labels.append(label)
            paac = list()
            for i in range(1, len(line)):
                paac.append(float(line[i]))
            data.append(paac)
    shuffle(data, labels, seed=30)
    return numpy.array(data), numpy.array(labels, dtype="float32")


def split_train_and_validation(split, data, labels):

    train_nu = int(len(labels) * split)
    val_nu = (len(labels) - train_nu) / 2
    train_data = data[:train_nu]
    train_label = labels[:train_nu]

    val_data = data[train_nu:][0:val_nu]
    val_label = labels[train_nu:][0:val_nu]

    test_data = data[train_nu:][val_nu:]
    test_label = labels[train_nu:][val_nu:]

    return train_data, train_label, val_data, val_label, test_data, test_label


def model(
        train_data, train_label, val_data, val_label, test_data, test_label):
    # set parameters:
    max_features = 50000
    batch_size = 64
    embedding_dims = 100
    nb_filters = 250
    hidden_dims = 250
    nb_epoch = 12

    # pool lengths
    pool_length = 2
    # level of convolution to perform
    filter_length = 3

    # length of APAAC
    maxlen = 20 + 2 * LAMBDA

    test_label_rep = test_label

    # Convert labels to categorical
    nb_classes = max(train_label) + 1
    train_label = np_utils.to_categorical(train_label, nb_classes)
    val_label = np_utils.to_categorical(val_label, nb_classes)
    test_label = np_utils.to_categorical(test_label, nb_classes)

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
    model.add(Dense(hidden_dims, nb_classes))
    model.add(Activation('sigmoid'))
    model.compile(
        loss='categorical_crossentropy', optimizer='adam')
    # weights_train = [1.0 if y == 1 else 1.0 for y in train_label]
    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_data=(val_data, val_label))
    score = model.evaluate(
        test_data, test_label,
        batch_size=batch_size, verbose=1, show_accuracy=True)
    print "Loss:", score[0], "Accuracy:", score[1]
    pred_data = model.predict_classes(test_data, batch_size=batch_size)
    print(classification_report(list(test_label_rep), pred_data))


if __name__ == '__main__':
    data, labels = load_data()
    split = 0.8
    data_train, labels_train, data_val, labels_val, data_test, labels_test = split_train_and_validation(split, data, labels)
    model(
        data_train, labels_train, data_val, labels_val, data_test, labels_test)
