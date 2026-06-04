
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics
import scripts.preprocessing.preprocessing as pre
import pylab as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#full, tied, diagonal, spherical


#silhouette graph
def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


def silhouette_graph(X, n_range, sample_size = 5000):
    n_clusters=np.arange(n_range[0], n_range[1])
    sils=[]
    sils_err=[]
    iterations=5
    
    for n in n_clusters:
        tmp_sil=[]
        for _ in range(iterations):
            # get random sub sample
            rand_idx = np.random.choice(len(X), sample_size)
            X_sub = X[rand_idx]
            # fit model and assign labe
            gmm_model = GMM(n_components=n, covariance_type="full")
            pipe = Pipeline([('scaler', StandardScaler()), ('gmm', gmm_model)])
            pipe.fit(X_sub) 
            labels=pipe.predict(X_sub)
            # compute silhouette score
            sil=metrics.silhouette_score(X_sub, labels, metric='euclidean')
            tmp_sil.append(sil)
        # get average silhouette score
        val=np.mean(SelBest(np.array(tmp_sil), int(iterations)))
        # get error bar
        err=np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)

    # plot graph
    fig, axis = plt.subplots(1, 1)
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("Number of clusters", fontsize = 14)
    plt.ylabel("Score", fontsize = 14)
    return fig
    


def run_sil_graph(keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], cluster_rng = [2, 10],
                            cm_num = 1):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    fig = silhouette_graph(input_arr[:, 0:len(keywords)], cluster_rng)
    keyword_str = '_'.join(keywords)
    fig.savefig(f"cluster_evaluations/silhouette_chart_{keyword_str}")

#run_sil_graph(["NH3", "PCld"], [ [0, 300], [1000, 3000]], [75, 105], [0, 200])
#run_sil_graph(["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], [75, 105], [0, 200])

#run_sil_graph(["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
  #                    [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], 
 #                     [75, 105], [0, 200], 4)