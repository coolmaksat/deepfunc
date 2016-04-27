#!/usr/bin/env python

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_fofe_sequence.py
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
    shuffle, encode_seq_one_hot, encode_seq, encode_seq_hydro
)
import os
import sys
import pdb
from keras.optimizers import Adam
from aaindex import (
    LUTR910101, NORM_AACOOC, AAINDEX,
    ALTS910101, CROG050101, OGAK980101,
    KOLA920101, TOBD000101, MEHP950102, RUSR970103)
import pandas as pd


MAXLEN = 500
DATA_ROOT = 'data/fofe/'


def load_data():
    train_df = pd.read_pickle(DATA_ROOT + 'train.pkl')
    # shuffle(df, seed=0)
    # numpy.random.seed(0)
    test_df = pd.read_pickle(DATA_ROOT + 'test.pkl')
    return train_df, test_df


def model(train_df, test_df):

    # Training
    batch_size = 64
    nb_epoch = 64

    train_data, test_data = train_df['data'].values, test_df['data'].values

    train_label, test_label = train_df['sequence'].values, test_df['sequence'].values
    for i in range(len(train_label)):
        train_label[i] = AAINDEX[train_label[i][-1]]

    for i in range(len(test_label)):
        test_label[i] = AAINDEX[test_label[i][-1]]
    train_label = np_utils.to_categorical(train_label, 20)
    test_label = np_utils.to_categorical(test_label, 20)

    # train_data = numpy.hstack(train_data).reshape(train_data.shape[0], 8000)
    # test_data = numpy.hstack(test_data).reshape(test_data.shape[0], 8000)

    print('X_train shape: ', train_data.shape)
    print('X_test shape: ', test_data.shape)
    model = Sequential()
    model.add(Dense(8000, activation='relu', input_dim=8000))
    model.add(Highway())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop')

    model_path = DATA_ROOT + 'fofe_sequence.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    model.fit(
        X=train_data, y=train_label,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopper])

    # Loading saved weights
    print 'Loading weights'
    model.load_weights(model_path)
    pred_data = model.predict_classes(
        test_data, batch_size=batch_size)
    return classification_report(list(test_label), pred_data)


def main(*args, **kwargs):
    train_df, test_df = load_data()
    report = model(train_df, test_df)
    print report


if __name__ == '__main__':
    main(*sys.argv)
