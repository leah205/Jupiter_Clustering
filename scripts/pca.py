import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_PCA(X, n_comp):
    pca = PCA(n_components = n_comp)
    pca.fit(X)
    # gets array of percentage of variance explained by each component
    print(pca.explained_variance_ratio_)

def get_pca_comp(X, var_threshold = 0.95):
    """
    Purpose
    ----------
    Finds significant principal component axes and returns reduced dimensional data

    Parameters
    --------
    X, 
    numpy array with axis 0 as pixels within lon/lat range and axis 1 as parameter pixel radiances

    var_threshold
    --------
    threshold for cumulative explained variance to choose number of components

    Returns
    ---------
    numpy array with axis 0 as pixels within lon/lat range and axis 1 as pixel radiances in new dimensions
    
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components = var_threshold)
    X_transformed = pca.fit_transform(X_scaled)
    # gets array of percentage of variance explained by each component
    print("variances")
    print(pca.explained_variance_ratio_)
    return X_transformed