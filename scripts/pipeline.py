
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plotting_processes as PLP
import scripts.clustering.clusters as CL 
import scripts.posterior_prob as PP
import scripts.cluster_stats as STAT
import scripts.pca as PCA
from dataclasses import dataclass, field
from config.config import cf
import config.types as TY
import scripts.plots.mapping as MP
import config.dicts as D
import json
# import regions






"""
functions to run various clustering --> visualization pipelines

 Parameters
    --------------------------------
    config, REQUIRED
        -instance of pipelineConfig
        
"""

def run_full_pipeline(config: TY.pipelineConfig):
    """
    Runs clustering according to parameters in config.cluster 
    and saves maps generated according to config.amp

    Parameters
    --------------
    config: instance of pipeline config class

    returns
    ------------
    void

    """
    print(config.cluster.n_comp)
    keywords = config.map.keywords
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]
    
    # function to cluster
    transform = CL.run_pca_pipeline if config.cluster.isPca else CL.run_raw_pipeline

    # get input numpy array and shape filtered by coordinates and radiance values
    [arr, subset_shape] = pre.get_input_array(config.map, param_ranges)

    indices = arr[:, arr.shape[1] - 1]
    data = arr[:, 0:len(keywords)]

    # run clustering function on processed data
    cluster_obj = transform(data, config.cluster)
    pred, probs, means, covariances = cluster_obj["pred"], cluster_obj["probs"], cluster_obj["means"], cluster_obj["covariances"]

    # map cluster outputs to original map shape (within coordinate range)
    reshaped_pred = MP.reshape_clustered(indices, subset_shape, pred)

    prefix = PLP.create_file_prefix(config.cluster, config.map)
    title = PLP.create_plot_title(config.cluster, config.map)
    
    if(len(keywords) == 2):
        # two-d radiance scatter plot
        plot_fig = PLP.create_plot_figure(config.map, pred, arr, reshaped_pred, title, cluster_obj, config.cluster.n_comp)
        plot_fig.savefig(f"{prefix}plot.png")
        # comparison of radiances in different filters mapped spatially
        map_comp_fig = PLP.create_map_comp_figure(config.map, reshaped_pred, title, config.cluster.n_comp)
        map_comp_fig.savefig(f"{prefix}map_comp.png")
    if(len(keywords) == 4):
        # two 2d radiance scatter plots
        plot_fig = PLP.create_plots_figure(config.map, pred, arr, reshaped_pred, title, config.cluster.n_comp)
        plot_fig.savefig(f"{prefix}plots.png")

    # heat map for mean value for each cluster for each parameter
    centroids_fig = STAT.get_centroids_figure(keywords, means, title)
    centroids_fig.savefig(f"{prefix}centroids.png")

    # spatial map of clusters
    map_fig = PLP.create_cluster_map(config.map, reshaped_pred, title, config.cluster.n_comp)
    map_fig.savefig(f"{prefix}cluster_map.png")
    keyword_str = '_'.join(keywords)
    
    stats =  STAT.get_all_stats(pred, indices, config.cluster.n_comp, config.map) 
    
    # first keyword is predictor, second is predictee
    if(len(config.map.keywords) == 2):
        stats = stats | STAT.get_cluster_regressions(data, probs, config.cluster.n_comp)
    
    stats["covariances"] = f"{covariances.tolist()}"
    print(stats)
   

    # dump means and standard deviations per cluster per parameter into json
    with open(f"{cf["json"]}") as f:
        data = json.load(f)
        dim_obj = (data
            .setdefault(config.map.name, {})
            .setdefault(keyword_str, {}))
        
        dim_obj[str(config.cluster.n_comp)] = stats

    with open(f"{cf["json"]}", "w") as f:
        
        json.dump(data, f, indent=2)
    

   # spatial maps for each cluster indicating probability the pixel belongs to that cluster
    uncertainty_fig = PP.create_uncertainty_fig(config.map, probs, indices, subset_shape, title)
    uncertainty_fig.savefig(f"{prefix}uncertainty_map.png")

    # spatial map indicating maximum posterior probability for each pixel
    max_prob_fig = PP.create_max_prob_map(config.map, probs, indices, subset_shape, title)
    max_prob_fig.savefig(f"{prefix}max_prob_fig.png")

    if(config.cluster.isPca):
        # gets the loadings of each cluster on each pca
        heat_map_fig = PCA.get_loadings_heatmap(cluster_obj["pca_obj"], keywords, title)
        heat_map_fig.savefig(f"{prefix}loadings.png")

        


# Run processes for longitude 0 - 30, latitude 90 - 105







"""
        keywords
        - substrings in radiance array file name to identify file (Ex: PCld)
        param_ranges
            - list of length two lists with desired values ranges for each dimension in clustering, same order as keywords
        latRng, OPTIONAL, default = [85, 95]
            - two element python list with minimum and maximum latitude specified
        lonRng, OPTIONAL, default = [230, 330]
            - two element python list with minimum and maximum longitude specified
        n_comp, OPTIONAL, default = 4
            - number of clusters for model
        cov_type, OPTIONAL, default = "full"
            - type of covariance for model
        cm_num, OPTIONAL, default = 1
            - number 1 - 3 specifying longitudinal mapping system
        threshold, OPTIONAL, default = 0.75
            - if soft clustering is enabled, specifies probability threshold for clustering
"""



# if __name__ == "__main__":

#     region = R.ROI_2
#     ROI = region["ROI"]


#     lon_range = [360 - region["lon_range"][1], 360 - region["lon_range"][0]]
#     lat_range = [90 - region["lat_range"][1], 90 - region["lat_range"][0]]

#     exp1Config = TY.pipelineConfig(
#         map = TY.mappingConfig(
#             keywords = ["NH3", "PCld"],
#             ROI = {},
#             latRng = lat_range,
#             lngRng = lon_range
#             source="2025"
#             ),
#         cluster = TY.clusterConfig(
#             n_comp = 4,
#         )
#     )


#     run_full_pipeline(exp1Config)











