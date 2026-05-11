
import numpy as np
import scripts.preprocessing.preprocessing as pre


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
    res = []
    for i in range(n_clusters):
        cluster_mask = cluster_arr == i
       
        # contains indices of selected cluster
        cluster_indices = indices[cluster_mask]

    
        clustered_dim = dim_arr[cluster_indices]
        mean = np.mean(clustered_dim)
        std = np.std(clustered_dim)
        print(f"{dim_name}: Cluster {i} mean: {mean:.2f}, std: {std:.2f} ")
        res.append([mean, std ])
    return res

def get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num):
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
    cluster_arr = np.array(pred)
    indices = indices.astype(np.int64)
    if(not (cluster_arr.shape == indices.shape) and cluster_arr.ndim == 1):
        raise TypeError("Index array length must match cluster array length")
    
    res = []
    for key in keywords:
        map =  pre.get_patch(key, latRng, lngRng, cm_num)
        dim_arr = map.flatten()
        res.append(get_stat(cluster_arr, dim_arr, indices, n_comp, key))
    return np.array(res)
    
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
    indices = np.argsort(cluster_means)
    return indices[pred]
    
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

    





#def get_all_cluster_stats(cluster_arr, NH3arr, pcldArr, CIarr, AOIarr, index_arr):

#def get_indice_cluster_stats(cluster_arr, ):
    
#def get_ammonia_cloud_cluster_stats()