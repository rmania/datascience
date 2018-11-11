## from the ass-kicking https://github.com/eriklindernoren/ML-From-Scratch

from itertools import combinations_with_replacement
import numpy as np
import math
import sys


def normalize(X, axis=-1, order=2):
    """
    Normalize the dataset X
    """
    ## atleast_1d --> inputs to arrays with at least one dimension
    ## np.linalg.norm --> Vector or matrix norm
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def standardize(X):
    """
    Standardize the dataset X
    """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    
    return X_std


def shuffle_data(X, y, seed=None):
    """ 
    Random shuffle of the samples in X and y
    """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ 
    Split the data into train and test sets 
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def make_diagonal(x):
    
    """
    Converts a vector into an diagonal matrix
    """
    m = np.zeros((len(x), len(x)))
    
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m

def calculate_covariance_matrix(X, Y=None):
    """ 
    Calculate the covariance matrix for the dataset X 
    """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)

