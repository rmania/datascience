import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN


def outliers_modified_z_score(series):  # noqa: F811 (Bas: Flake8 ignore for now, fix logic later)
    """
    uses the MAD and median as robust measures of central tendency and dispersion
    """
    threshold = 3.5

    median_y = np.median(series)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in series])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in series]
    return np.where(np.abs(modified_z_scores) > threshold)


def outliers_iqr(series):
    """
    Inter Quartile Range: range between 1st and 3rd quartile.
    Outliers are points falling outside of either 1.5 times the IQR
    """
    quartile_1, quartile_3 = np.percentile(series, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((series > upper_bound) | (series < lower_bound))


def is_outlier_std(group):
    lower_limit = group.mean() - (group.std() * 3)
    upper_limit = group.mean() + (group.std() * 3)
    return ~group.between(lower_limit, upper_limit)


def flag_outlier_pyod(df, feature_column, clf, outliers_fraction):

    """
    flags outlier using a PYOD ml model to identify outliers.
    info: https://pyod.readthedocs.io/en/latest/index.html
    Args
     df: pd dataframe
     feature_column: pd.Series that needs flagging
     clf: non-fitted pyod.model
     outliers_fraction: the proportion of outliers in the data set. Used when fitting to
     define the threshold on the decision function.
    """
    # initiate Pyod outlier classifiers

    scaler = MinMaxScaler()
    df[feature_column + "_scaled"] = scaler.fit_transform(df[[feature_column]])

    X = df[feature_column + "_scaled"].values.reshape(-1, 1)

    clf.fit(X)

    y_pred = clf.predict(X)
    logging.info(f"no flagged outliers: {y_pred[y_pred == 1].sum()}")

    return y_pred.tolist(), clf



def moving_average(data, window_size):
    """
    Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
     data: pd.Series independent variable
     window_size (int): rolling window size
    Returns:
     ndarray of linear convolution
    References:
     [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
     [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, "same")


def explain_anomalies(y, window_size, sigma=1.0):
    """
    Helps in exploring the anamolies using stationary standard deviation
    Args:
     y: pd.Series: independent variable
     window_size (int): rolling window size
     sigma (int): value for standard deviation
    Returns:
     dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
     containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {
        "standard_deviation": round(std, 3),
        "anomalies_dict": collections.OrderedDict(
            [
                (index, y_i)
                for index, y_i, avg_i in zip(
                    count(), y, avg  # noqa: F821 (Bas: Flake8 ignore for now, fix logic later)
                )
                if (y_i > avg_i + (sigma * std)) | (y_i < avg_i - (sigma * std))
            ]
        ),
    }


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """
    Helps in exploring the anamolies using rolling standard deviation
    Args:
     y: pd.Series: independent variable
     window_size (int): rolling window size
     sigma (int): value for standard deviation
    Returns:
     a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
     containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = (
        testing_std_as_df.replace(np.nan, testing_std_as_df.ix[window_size - 1]).round(3).iloc[:, 0].tolist()
    )
    std = np.std(residual)
    return {
        "stationary standard_deviation": round(std, 3),
        "anomalies_dict": collections.OrderedDict(
            [
                (index, y_i)
                for index, y_i, avg_i, rs_i in zip(
                    count(),  # noqa: F821 (Bas: Flake8 ignore for now, fix logic later)
                    y,
                    avg_list,
                    rolling_std,
                )
                if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))
            ]
        ),
    }


# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_outlier_results(
    x, y, window_size, sigma_value=1, text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False
):
    """
    Helps in generating the plot and flagging the anamolies.
    Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
    between the two.
    Args:
    x (pd.Series): dependent variable
    y (pd.Series): independent variable
    window_size (int): rolling window size
    sigma_value (int): value for standard deviation
    text_xlabel (str): label for annotating the X Axis
    text_ylabel (str): label for annotatin the Y Axis
    applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    ax.plot(x, y_av, color="green")
    ax.set(xlim=(0, 1000), xlabel=(text_xlabel), ylabel=(text_ylabel))

    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anomaly = np.fromiter(events["anomalies_dict"].keys(), dtype=int, count=len(events["anomalies_dict"]))
    y_anomaly = np.fromiter(
        events["anomalies_dict"].values(), dtype=float, count=len(events["anomalies_dict"])
    )
    ax.plot(x_anomaly, y_anomaly, "r*", markersize=12)

    # add grid and lines and enable the plot
    ax.grid(True)