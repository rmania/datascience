import numpy as np
import math
import sys


def mean_squared_error(y_true, y_pred):
    """
    Returns the mean squared error between y_true and y_pred
    """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def root_mean_squared_error(y_true, y_pred):
    """
    Returns the root mean squared error between y_true and y_pred
    """
    mse = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    return rmse
