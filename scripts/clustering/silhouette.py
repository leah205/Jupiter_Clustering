
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


#silhouette graph
def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


def silhouette_graph(X):
    n_clusters=np.arange(3, 9)
    sils=[]
    sils_err=[]
    iterations=5
    
    for n in n_clusters:
        tmp_sil=[]
        for _ in range(iterations):
            rand_idx = np.random.choice(len(X), 20000)
            X_sub = X[rand_idx]
            print(X_sub)
            gmm_model = GMM(n_components=n, covariance_type="full")
            pipe = Pipeline([('scaler', StandardScaler()), ('gmm', gmm_model)])
            pipe.fit(X_sub) 
            labels=pipe.predict(X_sub)
            
            sil=metrics.silhouette_score(X_sub, labels, metric='euclidean')
            tmp_sil.append(sil)
        #val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
        val=np.mean(SelBest(np.array(tmp_sil), int(iterations)))
        err=np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    fig, axis = plt.subplots(1, 1)
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")
    return fig
    


def run_sil_graph(keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                            cm_num = 1):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    fig = silhouette_graph(input_arr[:, 0:len(keywords)])
    keyword_str = '_'.join(keywords)
    fig.savefig(f"cluster_evaluations/silhouette_chart_{keyword_str}")

#run_sil_graph(["NH3", "PCld"], [ [0, 300], [1000, 3000]], [75, 105], [0, 200])
run_sil_graph(["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0, 1], [0, 1]], [75, 105], [0, 200])