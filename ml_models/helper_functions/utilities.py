import numpy as np


def check_valid_array(X, y, y_int=True):
    """
    checks if the array is a valid Numpy array
    Args
     X : aray of the feature set
     y : array of the target set
     y_int : checks if y.astype(np.integer)
    """
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

        
def reduce_mem_usage(df, verbose=True):
    """
    reduce memory usage by catting dtypes to types that need less internal memory
    args:
     df: Pandas Dataframe
     verbose: True describe memory usage reduction
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df
        
# Plotting utility functions 
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