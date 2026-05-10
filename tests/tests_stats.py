
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

def quantify_cluster(stats, param_ranges):
    param_swapped = np.swapaxes(param_ranges, 0, 1)
    normed_stats = (stats - param_swapped[0]) / (param_swapped[1] - param_swapped[0])
    print(stats - param_swapped[0])
    print(normed_stats.shape)
    return np.mean(normed_stats, axis = 1)

def test_quantify_cluster():
    stats = [[[5, 7], [0.2, 5]], [[3, 4], [0.5, 3]]]
    stats = np.array(stats)
    mean1 = ((3 / 8) + 0.2) / 2
    mean2 = ((1 / 8) + 0.5) / 2
    res = np.array([mean1, mean2])
    param_ranges = np.array([[2, 10], [0, 1]])
    np.testing.assert_array_equal(res, quantify_cluster(stats[:, :, 0], param_ranges))


def reassign_clusters(pred, stats, param_ranges):
    cluster_means = quantify_cluster(stats, param_ranges)
    print(cluster_means)
    indices = np.argsort(cluster_means)
    print(indices)
    return indices[pred]
    
def test_reassign_cluster():
    stats = [[[5, 7], [0.2, 5]], [[3, 4], [0.5, 3]]]
    stats = np.array(stats)
    mean1 = ((3 / 8) + 0.2) / 2
    mean2 = ((1 / 8) + 0.5) / 2
    res = np.array([mean1, mean2])
    param_ranges = np.array([[2, 10], [0, 1]])
    np.testing.assert_array_equal(res, quantify_cluster(stats[:, :, 0], param_ranges))
    pred = np.array([0, 1, 1, 0])
    res = np.array([1, 0, 0, 1])
    np.testing.assert_array_equal(res, reassign_clusters(pred, np.flip(stats[:, :, 0], axis = 0), param_ranges))





def test():
    x = np.arange(6)
    x1 = x.reshape((3, 2))
    y1, y2 = np.hsplit(x1, [1])
    np.testing.assert_equal(y1, [[0], [2], [4]])