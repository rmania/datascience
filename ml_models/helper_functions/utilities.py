import numpy as np

def check_valid_array(X, y, y_int=True):

    # check if ndarray
    if not isinstance(X, np.ndarray):
        raise ValueError('X must be a nparray. Found %s' % type(X))
    if not isinstance(y, np.ndarray):
        raise ValueError('y must be a nparray. Found %s' % type(y))

    if 'int' not in str(y.dtype):
        raise ValueError('y must be an integer array. Found %s. '
                         'Try passing the array as y.astype(np.integer)'
                         % y.dtype)

    if not ('float' in str(X.dtype) or 'int' in str(X.dtype)):
        raise ValueError('X must be an integer or float array. Found %s.'
                         % X.dtype)

    # check dim
    if len(X.shape) != 2:
        raise ValueError('X must be a 2D array. Pls reshape.Found %s' % str(X.shape))
    if len(y.shape) > 1:
        raise ValueError('y must be a 1D array.Pls reshape.Found %s' % str(y.shape))

    # check other
    if y.shape[0] != X.shape[0]:
        raise ValueError('y and X must contain the same number of samples. '
                         'Got y: %d, X: %d' % (y.shape[0], X.shape[0]))

