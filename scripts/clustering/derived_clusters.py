
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scripts.preprocessing.preprocessing import get_pix_arr, filter_pix_in_range
from sklearn.model_selection import GridSearchCV
import pandas as pd
#full, tied, diagonal, spherical

def gmm_bic_score(estimator, X):
    return estimator.bic(X)

param_grid = {
    "n_components": range(5, 10),
    "covariance_type": ["tied", "full", "diag", "spherical"]
}

grid_search = GridSearchCV(GMM(), param_grid = param_grid, scoring = gmm_bic_score)

X = get_pix_arr([[60, 160], [1600, 2000]], ["NH3", "PCld"])
grid_search.fit(X)


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

print(df.sort_values(by="BIC score").head())


