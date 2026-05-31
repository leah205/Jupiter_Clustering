
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plotting_processes as PLP
import scripts.clustering.clusters as CL 
import scripts.posterior_prob as PP
import scripts.cluster_stats as STAT
import scripts.pca as PCA
from dataclasses import dataclass, field
import config.config as cf




"""
functions to run various clustering --> visualization pipelines

 Parameters
    --------------------------------
    config, REQUIRED
        -instance of pipelineConfig
        
"""

def run_full_pipeline(config):
    param_ranges = [ranges_dict.get(keyword, [0, 1]) for keyword in config.keywords]
   
    
    transform = CL.run_pca_pipeline if config.isPca else CL.run_raw_pipeline

    [arr, subset_shape] = pre.get_input_array(config.keywords, param_ranges, config.latRng, config.lngRng, config.cm_num)

    indices = arr[:, arr.shape[1] - 1]
    data = arr[:, 0:len(config.keywords)]
    cluster_obj = transform(data, config.cov_type, config.n_comp, config.threshold)
    pred, probs, means = cluster_obj["pred"], cluster_obj["probs"], cluster_obj["means"]

    prefix = PLP.create_file_prefix(config.keywords, config.latRng, config.lngRng, config.n_comp, config.cm_num, config.threshold, config.isPca)
    
    if(len(config.keywords) == 2):
        plot_fig = PLP.create_plot_figure(config, pred, arr, subset_shape)
        plot_fig.savefig(f"{prefix}plot.png")
        map_comp_fig = PLP.create_map_comp_figure(config, pred, arr, subset_shape, param_ranges)
        map_comp_fig.savefig(f"{prefix}map_comp.png")
    if(len(config.keywords) == 4):
        plot_fig = PLP.create_plots_figure(config, pred, arr, subset_shape)
        plot_fig.savefig(f"{prefix}plots.png")
    centroids_fig = STAT.get_centroids_figure(config.keywords, means)
    centroids_fig.savefig(f"{prefix}centroids.png")
    map_fig = PLP.create_cluster_map(config, pred, arr, subset_shape)
    map_fig.savefig(f"{prefix}cluster_map.png")
    uncertainty_fig = PP.create_uncertainty_fig(config, probs, indices, subset_shape)
    uncertainty_fig.savefig(f"{prefix}uncertainty_map.png")
    max_prob_fig = PP.create_max_prob_map(config, probs, indices, subset_shape)
    max_prob_fig.savefig(f"{prefix}max_prob_fig.png")

    if(config.isPca):
        heat_map_fig = PCA.get_loadings_heatmap(cluster_obj["pca_obj"], config.keywords)
        heat_map_fig.savefig(f"{prefix}loadings.png")

        


# Run processes for longitude 0 - 30, latitude 90 - 105

ranges_dict = {
    "NH3": [0, 300],
    "PCld": [1000, 3100],
    "AOI": [0.1, 0.4],
    "CI": [0.3, 0.8],
}


ROI={"Hot Spot":[82,83,14.0,2.0],
                     "Gyre":[84,86,15.0,3.0],
                     "Cloud Plume":[82,84,5.0,3.0],
                     "Reference":[76,78,15,4.0]} 
lon_range = [0, 30]
lat_range = [0, 15]

lon_range = [360 - lon_range[1], 360 - lon_range[0]]
lat_range = [90 - lat_range[1], 90 - lat_range[0]]



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
@dataclass
class pipelineConfig:
    keywords: list[str]
    latRng: list[int]
    lngRng: list[int]
    n_comp: int
    ROI: dict = field(default_factory=dict)
    cov_type: str = "full"
    cm_num: int = 1
    threshold: float = 0.75
    isPca: bool = False


exp1Config = pipelineConfig(
    latRng = lat_range,
    lngRng = lon_range, 
    keywords = ["NH3", "PCld"],
    n_comp = 6
)

exp2Config = pipelineConfig(
    latRng = lat_range,
    lngRng = lon_range, 
    keywords = ["NH3", "PCld", "AOI", "CI"],
    n_comp = 6,
    ROI = ROI,
    threshold = 0.75
)

exp3Config = pipelineConfig(
    latRng = lat_range,
    lngRng = lon_range, 
    keywords = ["275", "395", "502", "619", "631", "645", "673", "727", "889"],
    n_comp = 5,
    ROI = ROI,
    isPca = True,
    threshold = 0.9)


run_full_pipeline(exp2Config)


