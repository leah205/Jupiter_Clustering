
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
#full, tied, diagonal, spherical
print("yo")
def create_bic_plot(pix_arr, n_range = [2, 10], sample_size = 10000):
  
    n_cluster = np.arange(n_range[0], n_range[1])
    bics = []
    bic_errs = []
    iterations = 10
    for n in n_cluster:
        tmp_bic = []
        for _ in range(iterations):
            X = pix_arr[np.random.choice(len(pix_arr), sample_size)]
            gmm_model = GMM(n_components=n, covariance_type="full")
            scaled = StandardScaler().fit(X).transform(X)
            gm = gmm_model.fit(scaled)
            bic = gm.bic(scaled)
            tmp_bic.append(bic)
        val =  np.mean(tmp_bic)
        err = np.std(tmp_bic)
        bics.append(val)
        bic_errs.append(err)

    fig, ax = plt.subplots(1, 1)

    plt.errorbar(n_cluster,bics, yerr=bic_errs, label='BIC')
    plt.title("BIC Scores", fontsize=20)
    plt.xticks(n_cluster)
    plt.xlabel("Number of clusters", fontsize = 14)
    plt.ylabel("Score", fontsize = 14)
    plt.legend()
    return fig


def run_bic(subdir, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], cluster_rng = [2, 10],
                            cm_num = 1):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pix_arr = input_arr[:, 0: len(keywords)]
    fig = create_bic_plot(pix_arr, cluster_rng)
    keyword_str = "_".join(keywords)
    fig.savefig(f"cluster_evaluations/{subdir}/BIC_plot_{keyword_str}")

lon_range = [0, 30]
lat_range = [0, 15]

lon_range = [360 - lon_range[1], 360 - lon_range[0]]
lat_range = [90 - lat_range[1], 90 - lat_range[0]]
cluster_rng = [2,16]

#run_bic("20251016UTc/lon_0-30", ["NH3", "PCld"], [[0, 300], [1000,  3000]], lat_range, lon_range)
#run_bic("20251016UTc/lon_0-30", ["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)
# run_bic("20251016UTc/lon_0-30", ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
#                  [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], lat_range, lon_range, cluster_rng, 1)

 

