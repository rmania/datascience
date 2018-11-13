import numpy as np

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