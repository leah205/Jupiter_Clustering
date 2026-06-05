
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plotting_processes as PLP
import scripts.clustering.clusters as CL 
import scripts.posterior_prob as PP
import scripts.cluster_stats as STAT
import scripts.pca as PCA
from dataclasses import dataclass, field
import config.config as cf
import config.types as TY
import scripts.plots.mapping as MP
import config.dicts as D




"""
functions to run various clustering --> visualization pipelines

 Parameters
    --------------------------------
    config, REQUIRED
        -instance of pipelineConfig
        
"""

def run_full_pipeline(config: TY.pipelineConfig):
    keywords = config.map.keywords
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]
    
    
    transform = CL.run_pca_pipeline if config.cluster.isPca else CL.run_raw_pipeline

    [arr, subset_shape] = pre.get_input_array(config.map, param_ranges)

    indices = arr[:, arr.shape[1] - 1]
    data = arr[:, 0:len(keywords)]
    cluster_obj = transform(data, config.cluster)
    pred, probs, means = cluster_obj["pred"], cluster_obj["probs"], cluster_obj["means"]
    reshaped_pred = MP.reshape_clustered(indices, subset_shape, pred)

    prefix = PLP.create_file_prefix(config.cluster, config.map)
    title = PLP.create_plot_title(config.cluster, config.map)
    
    if(len(keywords) == 2):
        plot_fig = PLP.create_plot_figure(config.map, pred, arr, reshaped_pred, title, cluster_obj)
        plot_fig.savefig(f"{prefix}plot.png")
        map_comp_fig = PLP.create_map_comp_figure(config.map, reshaped_pred, title)
        map_comp_fig.savefig(f"{prefix}map_comp.png")
    if(len(keywords) == 4):
        plot_fig = PLP.create_plots_figure(config.map, pred, arr, reshaped_pred, title)
        plot_fig.savefig(f"{prefix}plots.png")
    centroids_fig = STAT.get_centroids_figure(keywords, means, title)
    centroids_fig.savefig(f"{prefix}centroids.png")
    map_fig = PLP.create_cluster_map(config.map, reshaped_pred, title)
    map_fig.savefig(f"{prefix}cluster_map.png")

   
    uncertainty_fig = PP.create_uncertainty_fig(config.map, probs, indices, subset_shape, title)
    uncertainty_fig.savefig(f"{prefix}uncertainty_map.png")
    max_prob_fig = PP.create_max_prob_map(config.map, probs, indices, subset_shape, title)
    max_prob_fig.savefig(f"{prefix}max_prob_fig.png")

    if(config.cluster.isPca):
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



# region = R.ROI_2

# ROI = region["ROI"]
# lon_range = [360 - region["lon_range"][1], 360 - region["lon_range"][0]]
# lat_range = [90 - region["lat_range"][1], 90 - region["lat_range"][0]]

exp1Config = TY.pipelineConfig(
    map = TY.mappingConfig(
        keywords = ["275", "395", "502", "619", "631", "645", "673", "727", "889"],
        ROI = ROI,
        latRng = lat_range,
        lngRng = lon_range
        ),
    cluster = TY.clusterConfig(
        n_comp = 4,
      

    )
)


run_full_pipeline(exp1Config)


