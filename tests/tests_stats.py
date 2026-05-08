
import numpy as np

from statistics import mean

def get_cluster_stat(cluster_arr, dim_arr, indices, n_clusters):
    '''
        Purpose: Takes a cluster and outputs the mean and standard deviation of that cluster in a certain dimension
    '''
    # skip -1
    res = []
    for i in range(n_clusters):
        cluster_mask = cluster_arr == i
        # contains indices of selected cluster
        cluster_indices = indices[cluster_mask]
        clustered_dim = dim_arr[cluster_indices]
        
        res.append([np.mean(clustered_dim), np.std(clustered_dim)])
    return res

def test_get_cluster_stat():
    dim_arr = np.array([1, 2, 5, 0, 3, 0])
    cluster_arr = np.array([1, 1, 0, -1, 2])
    indices = np.array([0, 2, 3, 4, 5])
    n_clusters = 3
    cluster_stats = get_cluster_stat(cluster_arr, dim_arr, indices, n_clusters)

    # cluster stats should have length 3
    assert len(cluster_stats) == 3
    # cluster stats 
    [mean0, sd0] = cluster_stats[0]
    [mean1, sd1] = cluster_stats[1]
    [mean2, sd2] = cluster_stats[2]
    assert mean0 == 0
    assert mean1 == 3
    assert mean2 == 0



def test():
    x = np.arange(6)
    x1 = x.reshape((3, 2))
    y1, y2 = np.hsplit(x1, [1])
    np.testing.assert_equal(y1, [[0], [2], [4]])