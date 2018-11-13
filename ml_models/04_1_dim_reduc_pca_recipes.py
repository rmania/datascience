# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/principal_component_analysis.py

import numpy as np
import sys
sys.path.insert(0, 'helper_functions/')
from data_manipulation import calculate_covariance_matrix

class PCA():
    """
    Method for doing dim reduction by transforming the feature
    space to a lower dim removing correlation between features and
    maximizing the variance along each feature axis. 
    """
    def transform(self, X, n_components):
        """ 
        Fit dataset to the number of PCs specified in the
        constructor and return the transformed dataset
        Args
        ----------
        X : standardized X_train/X_test set
        n_components: number of eigen_pairs you wish to keep
        """
        covariance_matrix = calculate_covariance_matrix(X)

        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed