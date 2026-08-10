
import numpy as np
import scripts.preprocessing.preprocessing as pre
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import cf
from sklearn.linear_model import LinearRegression
import csv


keyword_dict = {
    "NH3": "Ammonia Mole Fraction (ppm)",
    "PCld": "Cloud Pressure (mb)",
    "AOI": "Altitude Opacity Index",
    "CI": "Color Index"
}

def get_stat(cluster_arr, dim_arr, indices, n_clusters, dim_name):
    '''
        Purpose: Takes a cluster and outputs the mean and standard deviation of that cluster in a certain dimension

        Parameters
        ----------------
        cluster_arr, TYPE 1D numpy array
            Cluster Array with cluster designations 0 - n componenets, -1 for no clustering assigned
        dim_arr, TYPE 1D numpy array
            spatially subsetted and flattenedarray of radiances for dimension
        indices, TYPE 1D numpy array of integers
            indices of clusters in cluster_arr
        n_clusters, TYPE Integer
            number of clusters
        Returns
        --------------
        List of mean and standard deviation pairs of each cluster in the dimension
    '''
    # skip -1
    res = {}
    for i in range(n_clusters):
        cluster_mask = cluster_arr == i
        # contains indices of selected cluster
        cluster_indices = indices[cluster_mask]

    
        clustered_dim = dim_arr[cluster_indices]
        mean = np.mean(clustered_dim)
        std = np.std(clustered_dim)
        print(f"{dim_name}: Cluster {i} mean: {mean:.2f}, std: {std:.2f} ")
        res[f"{i + 1}"] = {
            "mean": str(mean),
            "standard deviation": str(std)
        } 
       
    return res

def get_cluster_regressions(data, probs, n_clusters):
    """
   Gets object containing  cluster regressions for all clusters
   
    Parameters
    ----------------
    data: np array
        - radiance data for pixels
    probs: np array
        - two d array of probabilities that each pixel belongs to each cluster (pixels on axis 0)
    n_cluster: int
        - number of clusters

    Returns 
    ----------------
    dictionary containing entries of form coeff_n and b_n, characterizing linear regressions for each cluster n


   """
    res = {}
    for i in range(n_clusters):
        res = res | run_cluster_regression(data, probs, i)
    return res

def run_cluster_regression(data, probs,  i):
    """

    Runs linear regression on specific cluster

    Parameters
        ----------------
        data: np array
            - radiance data for pixels
        probs: np array
            - two d array of probabilities that each pixel belongs to each cluster (pixels on axis 0)
        i: int
            - cluster to run regression
    Returns
    ------------------

    dict with keys coeff_i and b_i for linear regression outputs

  
    """

    # covariances, and linear regression for NH3/pcld
    # get correlation
    # assumes nh3 is index 0, pcld at index 1
    # only runs if two dimensional
  
    res = {}
    x, y = data[:, 0].reshape(-1, 1), data[:, 1]
  
    m = LinearRegression()
    m.fit(
            x, y, sample_weight=probs[:, i]
    )
   
    #coeff_list = list(np.round(m.coef_, 3))
    # coeoff_strs = [f"{:.3f}".format(coeff) for coeff in coeff_list]
    return {f"coeff_{i + 1}": f"{m.coef_[0]:.3f}", f"b_{i + 1}": f"{m.intercept_:.3f}" }
      

    return res


def get_all_stats(pred, indices, n_comp, m):
    '''
    Purpose: Preprocesses data and passes it into a function to generate list of means and standard deviations for the clusters

    Parameters
    ----------------
    cluster_arr, TYPE python list of clusters
    dim_arr, TYPE 1D numpy array
    indices, TYPE 1D numpy array of floats
    n_clusters, TYPE Integer

    Returns
    --------------
    List of Lists of mean and standard deviation pairs for each cluster for each dimension


    
    '''
    dir_path = f"{cf["input"]}/{m.source}/{cf["sys"]}"

    cluster_arr = np.array(pred)
    indices = indices.astype(np.int64)
    if(not (cluster_arr.shape == indices.shape) and cluster_arr.ndim == 1):
        raise TypeError("Index array length must match cluster array length")
    
    res = {}
    for key in m.keywords:
        map =  pre.get_patch(key, m.latRng, m.lngRng, m.cm_num, dir_path)
        dim_arr = map.flatten()
        res[key] = get_stat(cluster_arr, dim_arr, indices, n_comp, key)

    
    return res
    
def reassign_clusters(pred, stats, param_ranges):
    """
    Parameters
    -----------
    pred
        np array of cluster assignments
    stats
        np array of means with axis 0 as clusters and axis 1 as dimension
    
    
    """
    
    cluster_means = quantify_clusters(stats, param_ranges)
    res = np.full(pred.shape, -1)
   
    unique, counts = np.unique(pred, return_counts = True)
 
    mask = ~(pred == -1)
    indices = np.argsort(cluster_means)
    res[mask] = indices[pred[mask]]
    unique, counts = np.unique(res, return_counts = True)
   
    return res

def quantify_clusters(stats, param_ranges):
    # convert param_ranges to numpy array
    """
    Purpose:  Find the mean of all means of clusters in all dimensions 

    Parameters:
    stats - nd array, first axis is dimension, second axis, is cluster means for that dimension
    param_ranges - similiarly ordered cross-dimensional array with min and max value for each dimension
    """

    param_swapped = np.swapaxes(param_ranges, 0, 1)
    normed_stats = (stats - param_swapped[0]) / (param_swapped[1] - param_swapped[0])
    return np.mean(normed_stats, axis = 1)

    
def get_centroids_figure(keywords, cluster_centroids, title):
    """
    Purpose
    --------------
    Gets a table of the means of each cluster in each original dimension

    Parameters
    --------------
    keywords
        array of dimension names
    cluster_centroid
        2D np array with axis 0 as clusters, axis 1 as dimensions (same order as keywords)

    Returns
    --------------
    Centroids table figure for each cluster and dimension

    """
    fig, ax = plt.subplots(1, 1)

    #check if dimension is wavelength (valid number)
    if(keywords[0].isdigit()):
        ax.set_title("Mean Cluster Reflectance per Wavelength")
    else:
        ax.set_title("Mean Cluster Value per Parameter")
    c_num = cluster_centroids.shape[0]
    cluster_labels = np.linspace(1, c_num, c_num).astype(np.int64)
    sns.heatmap(cluster_centroids, ax = ax, annot = True, cmap = "coolwarm", xticklabels = keywords, yticklabels = cluster_labels )
    fig.suptitle(title, fontsize = 10)
    return fig


