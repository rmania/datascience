import numpy as np
import sys
sys.path.insert(0, 'helper_functions/')
from data_manipulation import calculate_covariance_matrix

## --- https://sebastianraschka.com/notebooks/ml-notebooks.html

def compute_mean_vectors(X, y):
    """
    calculating the mean vectors for each class
    """
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
        
    print ('Mean vectors {} : {}\n'.format(label, mean_vectors[label - 1]))
    return mean_vectors
        
def within_class_scatter_matrix(X, y, print_result=None):
    """
    Constuct the Within-class Scatter Matrix
    """
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    
    S_W = np.zeros((n_features, n_features))
    
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))                 
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1) 
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat   
        
        if print_result:
            print ('Within-class Scatter Matrix:{}\n {}'.format(S_W.shape, S_W))
            
    return S_W

def between_class_scatter_matrix(X, y, print_result=None):
    """
    Construct the Between-class Scatter Matrix
    """
    overal_average = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = compute_mean_vectors(X, y)    
    
    S_B = np.zeros((n_features, n_features))
    
    for i, mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1) 
        overall_mean = overal_average.reshape(n_features, 1) 
        S_B += n * (mean_vec - overal_average).dot((mean_vec - overal_average).T)
        
        if print_result:
            print ('Between-class Scatter Matrix:{}\n {}'.format(S_B.shape, S_B))
            
    return S_B


def get_components(eig_vals, eig_vecs, n_comp=2):
    
    n_features = X.shape[1]
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(4, 1) for i in range(0, n_comp)])
    
    return W

## -----------------
class LDA():
    """
    source: https://github.com/eriklindernoren/ML-From-Scratch
    """
    def __init__(self):
        self.w = None

    def transform(self, X, y):
        self.fit(X, y)
        # Project data onto vector
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self, X, y):
        # Separate data by class
        X1 = X[y == 0]
        X2 = X[y == 1]

        # Calculate the covariance matrices of the two datasets
        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        # Calculate the mean of the two datasets
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred