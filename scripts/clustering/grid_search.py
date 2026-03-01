
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scripts.preprocessing.preprocessing import get_pix_arr
from sklearn.model_selection import GridSearchCV
import pandas as pd
#full, tied, diagonal, spherical

def gmm_bic_score(estimator, X):
    return estimator.bic(X)

def get_optimal_gmm_model(
                    pix_arr,
                    com_range = [1, 10], 
                      cov_types = ["tied", "full", "diag", "spherical"],
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
    
    param_grid = {
        "n_components": range(com_range[0], com_range[1]),
        "covariance_type": cov_types
    }

    grid_search = GridSearchCV(GMM(), param_grid = param_grid, scoring = gmm_bic_score)


    grid_search.fit(pix_arr)


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

pix_arr =  get_pix_arr([[60, 160], [1600, 2000]], ["NH3", "PCld"])
get_optimal_gmm_model(pix_arr, [5, 11])


