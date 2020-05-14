import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import logging


def stratifief_split(df, target_column: str, splits, plot_tscv=None):
    """
    timeseriessplit provides a moving and increasing training set and equal test size.
    """

    logging.info(f"dropping nan, creating X, y arrays")
    y = df.dropna()[target_column].values
    X = df.dropna().drop([target_column], axis=1).values
    logging.info(f"X shape {X.shape}, y shape : {y.shape}")

    skf = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(
            f"X_train : {X_train.shape} ,y_train shape: {y_train.shape},\
                    X_test shape: {X_test.shape}, y_test shape: {y_test.shape}"
        )
    if plot_tscv:
        f, ax = plt.subplots(figsize=(12, 5))
        for ii, (tr, tt) in enumerate(skf.split(X, y)):
            # Plot training and test indices
            l1 = ax.scatter(tr, [ii] * len(tr), c=[plt.cm.coolwarm(0.1)], marker="_", lw=6)
            l2 = ax.scatter(tt, [ii] * len(tt), c=[plt.cm.coolwarm(0.9)], marker="_", lw=6)
            _ = ax.set(
                ylim=[splits, -1],
                title="StratifiedKFold behavior",
                xlabel="data index",
                ylabel="CV iteration",
            )
            _ = ax.legend([l1, l2], ["Training", "Validation"])

    return skf, X, y, X_train, y_train, X_test, y_test, train_index, test_index