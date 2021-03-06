import numpy
import math
from sklearn.preprocessing import OneHotEncoder
from aaindex import (AALETTER, AAINDEX, HYDROPATHY_GROUPS)

encoder = OneHotEncoder()
hydro_encoder = OneHotEncoder()


def init_encoder():
    data = list()
    for l in AALETTER:
        data.append([ord(l)])
    encoder.fit(data)
    data = list()
    for l in [0, 1, 2]:
        data.append([l])
    hydro_encoder.fit(data)

init_encoder()


def encode_seq_one_hot(seq, maxlen=500):
    data = list()
    for l in seq:
        data.append([ord(l)])
    data = encoder.transform(data).toarray()
    data = list(data)
    data = data[:maxlen]
    while (len(data) < maxlen):
        data.append([0] * 20)
    return data


def encode_seq_hydro(seq, maxlen=500):
    data = list()
    for l in seq:
        data.append([HYDROPATHY_GROUPS[AAINDEX[l]]])
    data = hydro_encoder.transform(data).toarray()
    data = list(data)
    data = data[:maxlen]
    while (len(data) < maxlen):
        data.append([0] * 3)
    return data


def encode_seq(prop, seq, maxlen=500):
    data = list()
    for l in seq:
        data.append(prop[AAINDEX[l]])
    if isinstance(prop[0], list):
        while len(data) < maxlen:
            data.append([0.0] * 20)
    else:
        while len(data) < maxlen:
            data.append(0.0)
    return data


def train_val_test_split(labels, data, split=0.8, batch_size=16):
    """This function is used to split the labels and data
    Input:
        labels - array of labels
        data - array of data
        split - percentage of the split, default=0.8\
    Return:
        Three tuples with labels and data
        (train_labels, train_data), (val_labels, val_data), (test_labels, test_data)
    """
    n = len(labels)
    train_n = int((n * split) / batch_size) * batch_size
    val_test_n = int((n - train_n) / 2)

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    train = (train_labels, train_data)

    val_data = data[train_n:][0:val_test_n]
    val_labels = labels[train_n:][0:val_test_n]
    val = (val_labels, val_data)

    test_data = data[train_n:][val_test_n:]
    test_labels = labels[train_n:][val_test_n:]
    test = (test_labels, test_data)

    return (train, val, test)


def train_test_split(labels, data, split=0.8, batch_size=16):
    """This function is used to split the labels and data
    Input:
        labels - array of labels
        data - array of data
        split - percentage of the split, default=0.8\
    Return:
        Three tuples with labels and data
        (train_labels, train_data), (test_labels, test_data)
    """
    n = len(labels)
    train_n = int((n * split) / batch_size) * batch_size

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    train = (train_labels, train_data)

    test_data = data[train_n:]
    test_labels = labels[train_n:]
    test = (test_labels, test_data)

    return (train, test)


def shuffle(*args, **kwargs):
    """
    Shuffle list of arrays with the same random state
    """
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


def get_gene_ontology(filename='go.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open('data/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in go.keys():
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.iteritems():
        if 'children' not in val:
            val['children'] = list()
        for g_id in val['is_a']:
            if 'children' not in go[g_id]:
                go[g_id]['children'] = list()
            go[g_id]['children'].append(go_id)
    return go


def mean(a):
    """
    The mean value of the list data.
    Usage:
        result = mean(array)
    """
    return sum(a)/len(a)


def std(a, ddof=0):
    """
    The standard deviation of the list data.
    Usage:
        result = std(array)
    """
    m = mean(a)
    temp = [math.pow(i-m, 2) for i in a]
    res = math.sqrt(sum(temp) / (len(a) - ddof))
    return res


def normalize_aa(prop):
    """
    All of the amino acid indices are centralized and
    standardized before the calculation.
    Usage:
        result = normalize_aap(aap)
    Input: aap is a dict form containing the properties of 20 amino acids.
    Output: result is the a dict form containing the normalized properties
    of 20 amino acids.
    """

    if len(prop) != 20:
        raise Exception('Invalid number of Amino acids!')

    res = dict()
    aap_mean = mean(prop.values())
    aap_std = std(prop.values())
    for key, value in prop.iteritems():
        res[key] = (value - aap_mean) / aap_std
    return res
