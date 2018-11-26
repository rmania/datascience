import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
import scipy.stats as stats
import sys
sys.path.insert(0, 'helper_functions/')
from utilities import remove_borders

# settings
figsize=[12,6]


def normal_pdf(mean = 0.0, sd = .2, interval_width = .02):
    """
    Graph of normal probability density function, with flex intervals.
    Args:
     mean = mean of the distribution
     sd = standard deviation of distribution
     interval_width = interval width on x-axis
     remove_border = helper function to remove chart junk
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    xlow = mean - 3 * sd  # x-axis low end 
    xhigh = mean + 3 * sd  # x-axis high end
    
    x = np.arange(xlow, xhigh, interval_width)
    
    # Compute y values, i.e., probability density at each value of x:
    y = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-.5 * ((x - mean)/sd)**2)
    
    # plot
    ax.plot(x,y)
    ax.stem(x,y, markerfmt=' ')
    _ = ax.set(xlabel = '$x$', ylabel = '$p(x)$', title = 'Normal PDF')
    remove_borders(ax)
    
    # Approximate the integral as the sum of width * height for each interval.
    area = np.sum(interval_width * y)
    
    # text adjustments
    xadjust_plot = ((xhigh - xlow) / 7)
    yadjust_plot = ((y.max() - y.min()) / 9)
    
    _ = ax.text(xlow, y.max() - yadjust_plot, '$\mu$ = %s' % mean)
    _ = ax.text(xlow, y.max() - 2 * yadjust_plot, '$\sigma$ = %s' % sd)
    _ = ax.text(xhigh - xadjust_plot, y.max() - yadjust_plot, '$\Delta x$ = %s' % interval_width)
    _ = ax.text(xhigh - xadjust_plot, y.max() - 2 * yadjust_plot, '$\sum_{x}$ $\Delta x$ $p(x)$ = %5.3f' % area)
    
    return ax


def running_proportion_sim(N = 500, random_seed=None):
    """
    Toss a coin N times and compute running proportion of heads for simulation.
    Args:
     N = number of coin flips
     random_seed: put in seed (f.i. 123) for reproducability of results
    Example:
     ~.running_proportion_sim(N=10000, random_seed=False)
    """
    if random_seed:
        np.random.seed(random_seed)
        
    sequence = np.random.choice(a=(0,1), p=(.5, .5), size=N, replace=True)
    # running proportion of heads
    agg = np.cumsum(sequence)
    agg_total = np.linspace(1, N, N)
    proportion = agg/ agg_total
    
    fig, ax  = plt.subplots(figsize=figsize)
    
    ax.plot(agg_total, proportion, '--')
    remove_borders(ax)
    ax.set(xlabel = 'No of tosses', ylabel = 'Proportion heads', xlim = (1,N), 
           ylim = (0,1), title = 'Running Proportion of heads')
    ax.hlines(y=.5, xmin = agg.min(), xmax=agg.max(), linestyle='--', 
              color='red', alpha=.6)
    _ = ax.text(N/3, 0.2, 'End of run = %s' % proportion[-1])
    plt.xscale('log')
    
    return ax 

def running_posterior_probabilities_sim(n_trials = []):
    """
    Plot a sequence of updating posterior probabilities as we observe 
    increasing amounts of data.
    Args:
     n_trials: number of trials -> list with *even* number of trials 
    Example:
     _ = ~.running_posterior_probabilities_sim(n_trials=[0, 2, 4, 8, 15, 50, 250, 2500])
    """
    # plot
    fig, ax = plt.subplots(int(len(n_trials)/2), 2, sharex=True,figsize=[12,8])
    ax = ax.flatten()
    # config
    dist = stats.beta # beta continuous random variable.
    data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
    x = np.linspace(0, 1, 100)

    for k, N in enumerate(n_trials):
        heads = data[:N].sum()
        y = dist.pdf(x, 1 + heads, 1 + N - heads)
        ax[k].plot(x, y, label='{} tosses,\n {} heads'.format(N, heads))
        ax[k].fill_between(x, 0, y, color="#348ABD", alpha=0.4)
        ax[k].vlines(0.5, 0, 4, color="red", linestyles="--", lw=1)
        ax[k].set(xlabel = ('$p$, probability of heads') \
                           if k in [0, len(n_trials)-1] else ' ')
        ax[k].legend()
        remove_borders(ax[k])
        
    plt.tight_layout()
    
    return ax