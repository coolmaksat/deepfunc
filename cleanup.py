#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gen_next_level_data.py
'''

import numpy
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Flatten)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils import (
    shuffle, train_val_test_split,
    get_gene_ontology,
    get_model_max_features,
    encode_seq_one_hot)

import sys
import os
from collections import deque

LAMBDA = 24
DATA_ROOT = 'data/cnn/'
CUR_LEVEL = 'level_2/'
NEXT_LEVEL = 'level_3/'
MAXLEN = 1000


def get_model(
        go_id,
        parent_id,
        nb_filter=64,
        nb_row=3,
        nb_col=3,
        pool_length=2):
    filepath = DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.hdf5'
    model = Sequential()

    model.add(Convolution2D(nb_filter, nb_row, nb_col,
                            border_mode='valid',
                            input_shape=(1, MAXLEN, 20)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_length, pool_length)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    # Loading saved weights
    print 'Loading weights for ' + go_id
    model.load_weights(filepath)
    return model


def main(*args, **kwargs):
    if len(args) < 3:
        raise Exception('Please provide function id')
    parent_id = args[1]
    go_id = args[2]
    if len(args) == 4:
        level = int(args[3])
        global CUR_LEVEL
        global NEXT_LEVEL
        CUR_LEVEL = 'level_' + str(level) + '/'
        NEXT_LEVEL = 'level_' + str(level + 1) + '/'
    try:
        model = get_model(go_id, parent_id)
    except Exception, e:
        print e
        filepath = DATA_ROOT + CUR_LEVEL + parent_id + '/' + go_id + '.hdf5'
        print "Removing " + filepath
        os.remove(filepath)


if __name__ == '__main__':
    main(*sys.argv)
