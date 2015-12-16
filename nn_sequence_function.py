#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_sequence.py
'''

import os
import numpy
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils


os.environ.setdefault(
    'THEANO_FLAGS', 'mode=FAST_RUN,device=gpu,floatX=float32')

DETECTOR_LEN = 26
ACIDS_LEN = 26


def shuffle(*args, **kwargs):
    rng_state = numpy.random.get_state()
    for arg in args:
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(arg)


def get_motif_detector(seq):
    detector = numpy.zeros((ACIDS_LEN, DETECTOR_LEN), dtype='float32')
    max_value = 0
    for i in range(len(seq)):
        detector[ord(seq[i]) - ord('A')][i % DETECTOR_LEN] += 1.0
        if max_value < detector[ord(seq[i]) - ord('A')][i % DETECTOR_LEN]:
            max_value = detector[ord(seq[i]) - ord('A')][i % DETECTOR_LEN]
    detector /= max_value
    return detector


def load_data():
    data = list()
    labels = list()
    with open('data/obtained/go_sequence.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            labels.append(int(line[0]))
            detector = get_motif_detector(line[1])
            data.append(detector)
    shuffle(data, labels, random_state=0)
    return numpy.array(data), numpy.array(labels)


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
    # Parameters
    batch_size = 16
    nb_classes = 2
    nb_epoch = 48
    # shape of the image (SHAPE x SHAPE)
    shapex, shapey = ACIDS_LEN, DETECTOR_LEN
    # number of convolutional filters to use
    nb_filters = 32
    # level of pooling to perform (POOL x POOL)
    nb_pool = 2
    # level of convolution to perform (CONV x CONV)
    nb_conv = 3

    train_data = train_data.reshape(train_data.shape[0], 1, shapex, shapey)
    test_data = test_data.reshape(test_data.shape[0], 1, shapex, shapey)
    val_data = val_data.reshape(val_data.shape[0], 1, shapex, shapey)

    test_label_rep = test_label

    # train_label = np_utils.to_categorical(train_label, 2)
    # val_label = np_utils.to_categorical(val_label, 2)
    # test_label = np_utils.to_categorical(test_label, 2)

    model = Sequential()
    model.add(Convolution2D(
        nb_filters, 1, nb_conv, nb_conv, border_mode='full'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(
    #     nb_filters, nb_filters, nb_conv, nb_conv))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # the resulting image after conv and pooling is the original shape
    # divided by the pooling with a number of filters for each "pixel"
    # (the number of filters is determined by the last Conv2D)
    model.add(Dense(nb_filters * (shapex / nb_pool + 1) * (shapey / nb_pool + 1), 128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(128, 1))
    model.add(Activation('sigmoid'))
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
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
    split = 0.9
    data_train, labels_train, data_val, labels_val, data_test, labels_test = split_train_and_validation(split, data, labels)
    model(
        data_train, labels_train, data_val, labels_val, data_test, labels_test)
