#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn_sequence.py
"""

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Activation, Input, Flatten, merge)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (
    Convolution1D, MaxPooling1D)
from sklearn.metrics import classification_report
from utils import (
    shuffle,
    get_gene_ontology)
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
from aaindex import (
    AAINDEX)
from collections import deque

DATA_ROOT = 'data/yeast/'
MAXLEN = 500
go = get_gene_ontology('goslim_yeast.obo')


def get_go_set(go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set

functions = get_go_set('GO:0003674')
functions.remove('GO:0003674')
functions = list(functions)
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind


def load_data():
    train_df = pd.read_pickle(DATA_ROOT + 'train.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'test.pkl')

    train_data = list()
    train_labels = list()
    for i in range(len(functions)):
        train_labels.append(list())
    for row in train_df.iterrows():
        seq = row[1]['sequences']
        item = [0] * MAXLEN
        for i in range(min(MAXLEN, len(seq))):
            item[i] = AAINDEX[seq[i]]
        train_data.append(item)
        gos = row[1]['gos']
        label = [0] * len(functions)
        for go_id in gos:
            label[go_indexes[go_id]] = 1
        for i in range(len(label)):
            train_labels[i].append(label[i])
    test_data = list()
    test_labels = list()
    for i in range(len(functions)):
        test_labels.append(list())
    for row in test_df.iterrows():
        seq = row[1]['sequences']
        item = [0] * MAXLEN
        for i in range(min(MAXLEN, len(seq))):
            item[i] = AAINDEX[seq[i]]
        test_data.append(item)
        gos = row[1]['gos']
        label = [0] * len(functions)
        for go_id in gos:
            label[go_indexes[go_id]] = 1
        for i in range(len(label)):
            test_labels[i].append(label[i])
    return (
        (train_labels, np.array(train_data)),
        (test_labels, np.array(test_data)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def get_feature_model():
    embedding_dims = 128
    max_features = 20
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=0.2))
    model.add(Convolution1D(
        nb_filter=64,
        filter_length=16,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=8))
    model.add(Flatten())
    return model


def get_function_node(go_id, parent_models):
    if len(parent_models) == 1:
        dense = Dense(128, activation='relu')(parent_models[0])
    else:
        merged_parent_models = merge(parent_models, mode='concat')
        dense = Dense(128, activation='relu')(merged_parent_models)
    output = Dense(1, activation='sigmoid', name=go_id)(dense)
    return dense, output


def model():
    # set parameters:
    batch_size = 256
    nb_epoch = 100
    nb_classes = len(functions)
    train, test = load_data()
    train_labels, train_data = train
    test_labels, test_data = test

    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input')
    feature_model = get_feature_model()(inputs)
    go['GO:0003674']['model'] = feature_model
    q = deque()
    for go_id in go['GO:0003674']['children']:
        q.append(go_id)

    while len(q) > 0:
        go_id = q.popleft()
        parent_models = list()
        for p_id in go[go_id]['is_a']:
            parent_models.append(go[p_id]['model'])
        dense, output = get_function_node(go_id, parent_models)
        go[go_id]['model'] = dense
        go[go_id]['output'] = output
        for ch_id in go[go_id]['children']:
            q.append(ch_id)

    output_models = [None] * nb_classes
    for go_id, ind in go_indexes.iteritems():
        output_models[ind] = go[go_id]['output']

    model = Model(input=inputs, output=output_models)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model_path = DATA_ROOT + 'hierarchical.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    output = {}
    for i in range(len(functions)):
        output[functions[i]] = np.array(train_labels[i])
    model.fit(
        {'input': train_data},
        output,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopper])

    print 'Loading weights'
    model.load_weights(model_path)
    output_test = {}
    for i in range(len(functions)):
        output_test[functions[i]] = np.array(test_labels[i])
    score = model.evaluate({'input': test_data}, output_test)
    predictions = model.predict(test_data, batch_size=batch_size, verbose=1)
    for i in range(len(test_labels)):
        pred = np.round(predictions[i].flatten())
        test = test_labels[i]
        print functions[i]
        print classification_report(test, pred)
    print 'Test loss:\t', score[0]
    print 'Test accuracy:\t', score[1]


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    model()

if __name__ == '__main__':
    main(*sys.argv)
