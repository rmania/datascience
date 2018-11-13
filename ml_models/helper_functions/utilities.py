import numpy as np
import matplotlib.pyplot as plt

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


## Plotting utility functions 
        
def remove_borders(ax, left=False, bottom=False, right=True, top=True):
    """Remove chart junk from matplotlib plots.
    Args
    ----------
    axes : fi. plt.subplots()
    left : bool (default: `False`)
        Hide left axis spine if True.
    bottom : bool (default: `False`)
        Hide bottom axis spine if True.
    right : bool (default: `True`)
        Hide right axis spine if True.
    top : bool (default: `True`)
        Hide top axis spine if True.
    """
    
    ax.spines["top"].set_visible(not top)
    ax.spines["right"].set_visible(not right)
    ax.spines["bottom"].set_visible(not bottom)
    ax.spines["left"].set_visible(not left)
    if bottom:
        ax.tick_params(bottom=False, labelbottom=False)
    if top:
        ax.tick_params(top=False)
    if left:
        ax.tick_params(left=False, labelleft=False)
    if right:
        ax.tick_params(right=False)