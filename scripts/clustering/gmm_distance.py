import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import scripts.preprocessing.preprocessing as pre
import scripts.clustering.clusters as CL 
import pylab as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_js(m1, m2, n_sample = 10**5):
   
    x = m1.sample(n_sample)[0]
    log_1_x = m1.score_samples(x)
    log_2_x = m2.score_samples(x)
    log_mix_X = np.logaddexp(log_1_x, log_2_x)

    y = m1.sample(n_sample)[0]
    log_1_y = m1.score_samples(y)
    log_2_y = m2.score_samples(y)
    log_mix_Y = np.logaddexp(log_1_y, log_2_y)

    squared_dist = (log_1_x.mean() - (log_mix_X.mean() - np.log(2))
            + log_2_y.mean() - (log_mix_Y.mean() - np.log(2))) / 2
    return np.sqrt(np.maximum(0, squared_dist))

def create_js_plot(X, n_range, sample_size = 10000):
    n_clusters = np.arange(n_range[0], n_range[1])
  
    iterations = 10
    results = []
    res_errs = []
    
    for n in n_clusters:
        dist = []
        for _ in range(iterations):
            X_sub = X[np.random.choice(len(X), sample_size)]

            # create 50/50 data split
            x1, x2 = train_test_split(X_sub, test_size = 0.5)

            #fit models
            gmm_model = GMM(n_components=n, covariance_type="full")
            #pipe = Pipeline([('scaler', StandardScaler()), ('gmm', gmm_model)])
            #gmm_1 = pipe.fit(x1)
            #gmm_2 = pipe.fit(x2)

            scaled1 = StandardScaler().fit(x1).transform(x1)
            scaled2 = StandardScaler().fit(x2).transform(x2)
            gmm_1 = gmm_model.fit(scaled1)
            gmm_2 = gmm_model.fit(scaled2)
            dist.append(get_js(gmm_1, gmm_2))
        result = np.mean(np.array(dist))
        res_err = np.std(np.array(dist))
        results.append(result)
        res_errs.append(res_err)

    fig, ax = plt.subplots(1, 1)
    plt.errorbar(n_clusters, results, yerr = res_errs)
    plt.title("Difference Between GMMS")
    plt.xticks(n_clusters)
    plt.xlabel("Number of Clusters")
    plt.ylabel("GMM JS Distance")
    return fig


def run_js_dist(keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], cluster_rng = [2, 10],
                            cm_num = 1):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pix_arr = input_arr[:, 0: len(keywords)]
    fig = create_js_plot(pix_arr, cluster_rng)
    keyword_str = "_".join(keywords)
    fig.savefig(f"cluster_evaluations/GMM_distance_{keyword_str}")
    

#run_js_dist(["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], [75, 105], [0, 200])
#run_js_dist(["NH3", "PCld"], [[0, 300], [1000, 3000]], [75, 105], [0, 200])