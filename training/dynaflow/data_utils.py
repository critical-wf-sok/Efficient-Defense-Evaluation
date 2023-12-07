import os
import numpy as np
import re
from tqdm import tqdm
import math
import scipy
from scipy.stats import kstest
import random


def load_trace(fi, separator="\t", filter_by_size=False):
    """
    loads data to be used for predictions
    """
    sequence = [[], [], []]
    for line in fi:
        pieces = line.strip("\n").split(separator)
        if int(pieces[1]) == 0:
            break
        timestamp = float(pieces[0])
        length = abs(int(pieces[1]))
        direction = int(pieces[1]) // length
        if filter_by_size:
            # ignore packets whose size is less than a Tor Cell
            if length > 512:
                sequence[0].append(timestamp)
                sequence[1].append(length)
                sequence[2].append(direction)
        else:
            sequence[0].append(timestamp)
            sequence[1].append(length)
            sequence[2].append(direction)
    return sequence


def load_data(directory, separator='\t', 
        fname_pattern=r"(\d+)[-](\d+)", max_classes=99999, 
        min_instances=0, max_instances=500, open_world=False, **kwargs):
    """
    Load feature files from a directory.
    Parameters
    ----------
    directory : str
        System file path to a directory containing feature files.
    input_size : int
        The length of the vector used to represent each data sample.
    separator : str
        Character string used to split features in the feature files.
    max_classes : int
        Maximum number of classes to load.
    max_instances : int
        Minimum number of instances acceptable per class.
        If a class has less than this number of instances, the all instances of the class are discarded.
    max_instances : int
        Maximum number of instances to load per class.
    Returns
    -------
    ndarray
        Numpy array of Nxf containing site visit feature instances.
    ndarray
        Numpy array of Nx1 containing the labels for site visits.
    """
    # load pickle file if it exist
    times = []
    X = []  # feature instances
    Y = []  # site labels
    for root, dirs, files in os.walk(directory):

        # filter for trace files
        files = [fname for fname in files if re.match(fname_pattern, fname)]

        # read each feature file as CSV
        class_counter = dict()  # track number of instances per class
        for fname in tqdm(files, total=len(files)):

            # get trace class
            if not open_world:
                cls = int(re.match(fname_pattern, fname).group(1))
                cls //= 10
            else:
                cls = 0

            # skip if maximum number of instances reached
            if class_counter.get(cls, 0) >= max_instances:
                continue

            # skip if maximum number of classes reached
            if int(cls) >= max_classes:
                continue

            # load the trace file
            with open(os.path.join(root, fname), "r") as fi:
                trace = load_trace(fi, separator, filter_by_size=False)

                # use direction information to represent trace sample
                #rep = np.array(trace[2])
                #rep = np.array(trace[0])
                #times.append(rep[-1])
                #rep = np.array(trace[0])*np.array(trace[2])

                # create iat window-based representation
                rep = []
                for i in range(1,len(trace[0])):
                    rep.append(trace[0][i] - trace[0][i-1])
                rep = np.array([np.mean(a) for a in np.array_split(rep, 200)])
                # pad trace length with zeros
                if len(rep) < input_size:
                    rep = np.concatenate((rep, np.zeros((input_size - len(rep),))))
                rep = rep[:input_size]

                X.append(rep)
                Y.append(cls)
                class_counter[int(cls)] = class_counter.get(cls, 0) + 1

    ## trim data to minimum instance count
    counts = {y: Y.count(y) for y in set(Y)}
    new_X, new_Y = [], []
    for x, y in zip(X, Y):
        if counts[y] >= min_instances:
            new_Y.append(y)
            new_X.append(x)
    X, Y = new_X, new_Y

    # adjust labels such that they are assigned a number from 0..N
    # (required when labels are non-numerical or does not start at 0)
    # try to keep the class numbers the same if numerical
    labels = list(set(Y))
    labels.sort()
    d = dict()
    for i in range(len(labels)):
        d[labels[i]] = i
    Y = list(map(lambda x: d[x], Y))

    X, Y = np.array(X)[...,np.newaxis], np.array(Y)

    # convert array types to floats
    X = X.astype('float32')
    Y = Y.astype('float32')

    # return X and Y as numpy arrays
    return X, Y


def make_split(X, y, train_perc, test_perc):
    """
    Shuffle and split dataset into training, testing, and validation sets.
    """
    sample_count = len(y)
    classes = len(np.unique(y))

    # shuffle
    s = np.arange(sample_count).astype(np.int)
    np.random.seed(0)
    np.random.shuffle(s)
    X, y = X[s], y[s]

    # slice into train, test, and validation
    tr_cut = int(sample_count * train_perc)
    X_tr  = X[:tr_cut, :]
    y_tr  = y[:tr_cut]
    te_cut = int(sample_count * (train_perc+test_perc))
    X_te  = X[tr_cut:te_cut, :]
    y_te  = y[tr_cut:te_cut]
    X_va  = X[te_cut:, :]
    y_va  = y[te_cut:]
    return X_tr, y_tr, X_te, y_te, X_va, y_va
