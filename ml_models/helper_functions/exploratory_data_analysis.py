## simplified from mlxtend Machine Learning Library Extensions Sebastian Raschka


import matplotlib.pyplot as plt
import numpy as np

def ecdf(x, y_label='ECDF', x_label=None, ax=None, percentile=None,
             ecdf_marker='o'):
    """Plots an Empirical Cumulative Distribution Function
    Parameters
    ----------
    x : array or list, shape=[n_samples,]
        Array-like object containing the feature values
    y_label : str (default='ECDF') label for the y-axis
    x_label : str (default=None) label for the x-axis
    percentile : float (default=None)
        Float between 0 and 1 for plotting a percentile
        
    Returns
    ---------
    ax : matplotlib.axes.Axes object
    percentile_threshold : float
    percentile_count : Number of if percentile is not None
        Number of samples that have a feature less or equal than
        the feature threshold at a percentile threshold
        or None if `percentile=None`
    """
    fig, ax = plt.subplots(1,1, figsize=[10,6])
    
    x = np.sort(x)
    y = np.arange(1, x.shape[0] + 1) / float(x.shape[0])

    ax.plot(x, y, marker='o', linestyle='', color='steelblue')
    ax.set(ylabel = y_label, title = 'Empirical Cumulative Distribution Plot')
    
    if x_label is not None:
        ax.set_xlabel(x_label)
    
    if percentile:
        targets = x[y <= percentile]
        percentile_threshold = targets.max()
        percentile_count = targets.shape[0]
        ax.axvline(percentile_threshold,
                   color='red',
                   linestyle='--')
    else:
        percentile_threshold = None
        percentile_count = None

    return ax, percentile_threshold, percentile_count