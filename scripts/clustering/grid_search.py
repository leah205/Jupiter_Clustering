
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

def gmm_bic_score(estimator, X):
    return estimator.bic(X)

def get_optimal_gmm_model(
                    pix_arr,
                    com_range = [1, 10], 
                      cov_types = ["full"],
                       ):
    '''
    Performs grid search to find optimal covariance types and number of components for gmm model
    
    Parameters
    -----------
    pix_arr, TYPE NONOPTIONAL
    DESCRIPTION:  numpy array with axis 0 as pixels within lon/lat range and axis 1 as parameter pixel radiances

    com_range, TYPE OPTIONAL
    DESCRIPTION: length 2 array with min and max in clustering componenet range to be searched
    DEFAULT: [5, 10]

    cov_types, TYPE OPTIONAL
    DESCRIPTION: Array of covariance types for grid search
    DEFAULT:  ["tied", "full", "diag", "spherical"]

    Side effects
    -------------
    Prints five covariance/component combinations with lowest BIC score

    Returns
    ------------
    None
    '''
    
    print(com_range)
    param_grid = {
        "n_components": range(com_range[0], com_range[1]),
        "covariance_type": cov_types
    }

    #grid_search = GridSearchCV(GMM(), param_grid = param_grid, scoring = gmm_bic_score)
    grid_search = GridSearchCV(GMM(), param_grid = param_grid)
    print("fitting...")
    grid_search.fit(pix_arr)
    print("done!")


    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
        }
    )
    print(df.sort_values(by = "BIC score").head())



def run_grid_search(keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                            cm_num = 1):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    #pred1 = CL.create_clusters(input_arr[:, 0:len(keywords)], "full", 3, False, 0)

    #pred2 = CL.create_clusters(input_arr[:, 0:len(keywords)], "full", 4, False, 0)
    pred3 = CL.create_clusters(input_arr[:, 0:len(keywords)], "full", 5, False, 0)
    pred10 = CL.create_clusters(input_arr[:, 0:len(keywords)], "full", 10, False, 0)
    pred10 = CL.create_clusters(input_arr[:, 0:len(keywords)], "full", 20, False, 0)



#get_optimal_gmm_model(input_arr[:, 0:len(keywords)], [3, 3])
#run_grid_search(["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0, 1], [0, 1]], [75, 105], [0, 200])





