import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

    
        

def pprint_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prettyprint confusion matrix, as returned by sklearn.metrics.confusion_matrix, 
    as a heatmap.
    
    Args
    ---------
    confusion_matrix: numpy.ndarray (as returned from call to 
        sklearn.metrics.confusion_matrix. 
    class_names: list
        An ordered list of class names, in order of confusion matrix.
    figsize: tuple
    fontsize: int
        Font size for axes labels. Default: 14.
        
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot confusion matrix
    """
    
    
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    
    # config colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    cmap = mpl.cm.RdYlGn_r
    vmin = confusion_matrix.min()
    vmax = confusion_matrix.max()
    levels = MaxNLocator(nbins=10).tick_values(vmin, vmax)
        
    _ = cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, alpha=.7,
                                norm=None,                                
                                boundaries=levels,
                                ticks=levels,
                                spacing='proportional',
                                orientation='vertical')
    cb._A = []
    cb.set_label('label counts', size=10)
    
    try:
        hm = sns.heatmap(df_cm, annot=True, annot_kws={'color': 'black', 'size': 12}, 
                         cmap=cmap, fmt='d', alpha=.8, 
                         linewidths=1, ax=ax, cbar=ax==cb)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    _ = ax.set(ylabel = 'True label', xlabel = 'Predicted label')
    
    return ax, cax



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')