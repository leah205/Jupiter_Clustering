from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from numpy._typing._array_like import NDArray

import config.dicts as D
import config.types as TY
import scripts.clustering.clusters as CL
import scripts.fits as F
import scripts.helpers.io as IO
import scripts.pca.pca as PCA
import scripts.plots.mapping as MP
import scripts.plots.plotting_processes as PLP
import scripts.preprocessing.preprocessing as pre
import scripts.statistics.cluster_stats as STAT
import scripts.statistics.io as STAT_IO
import scripts.statistics.posterior_prob as PP

# import regions


"""
functions to run various clustering --> visualization pipelines

 Parameters
    --------------------------------
    config, REQUIRED
        -instance of pipelineConfig
        
"""


def run_full_pipeline(config: TY.pipelineConfig) -> None:
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

    output_dir = IO.get_output_dir(config.cluster, config.map)
    output_dir_path = Path(output_dir)
    # clustering has already been run
    if output_dir_path.exists():
        return
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(output_dir)

    # function to cluster
    transform = CL.run_pca_pipeline if config.cluster.isPca else CL.run_raw_pipeline

    input_obj = load_pipeline_inputs(config)

    # run clustering function on processed data
    cluster_obj: dict[str, NDArray[Any]] = transform(input_obj.data, config.cluster)

    results = prepare_cluster_results(cluster_obj, input_obj)

    prefix = output_dir_path / IO.create_file_prefix(
        config.cluster, config.map, input_obj.files[0]
    )

    hdul = F.create_cluster_fits(
        cluster_map=results.cluster_map,
        prob_map=results.prob_map,
        n_comp=config.cluster.n_comp,
    )
    hdul.writeto(f"{prefix}.fits")

    generate_visualizations(input_obj, results, config, output_dir)

    stats = STAT.get_all_stats(
        results=results,
        indices=input_obj.indices,
        n_comp=config.cluster.n_comp,
        map_config=config.map,
        data=input_obj.data,
    )
    # dump means and standard deviations per cluster per parameter into json
    STAT_IO.save_stats_json(config, stats)


def prepare_cluster_results(cluster_obj: dict, input_obj: TY.input_obj) -> TY.results:
    pred, probs = (
        cluster_obj["pred"],
        cluster_obj["probs"],
    )

    # map cluster outputs to original map shape (within coordinate range)
    pred_map = MP.reshape_clustered(pred, input_obj.indices, input_obj.map_shape)

    # clusters along first axis
    probs_swapped = np.swapaxes(probs, 0, 1)
    prob_map = np.apply_along_axis(
        MP.reshape_clustered,
        1,
        probs_swapped,
        input_obj.indices,
        input_obj.map_shape,
    )

    return TY.results(**cluster_obj, prob_map=prob_map, cluster_map=pred_map)


def load_pipeline_inputs(config: TY.pipelineConfig) -> TY.input_obj:
    keywords = config.map.keywords
    input_dir = IO.get_dir_path(config.map)
    fits_files = IO.get_fits_files(input_dir, keywords)
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]
    # get input numpy array and shape filtered by coordinates and radiance values
    [data, subset_shape, indices] = pre.get_input_array(
        config.map, param_ranges, fits_files
    )
    return TY.input_obj(
        data=data, map_shape=subset_shape, indices=indices, files=fits_files
    )


def generate_visualizations(
    input_obj: TY.input_obj,
    results: TY.results,
    config: TY.pipelineConfig,
    output_dir: str,
):
    title = PLP.create_plot_title(config.cluster, config.map)
    vis_dir_path = Path(output_dir + "/visualizations")
    vis_dir_path.mkdir(parents=True, exist_ok=True)

    v_prefix = vis_dir_path / IO.create_file_prefix(
        config.cluster, config.map, input_obj.files[0]
    )

    if len(config.map.keywords) == 2:
        # two-d radiance scatter plot
        plot_fig = PLP.create_plot_figure(
            config=config.map,
            arr=input_obj.data,
            title=title,
            result=results,
            n_comp=config.cluster.n_comp,
        )
        IO.saveFigure(Path(f"{v_prefix}plot.png"), plot_fig)
        # comparison of radiances in different filters mapped spatially
        map_comp_fig = PLP.create_map_comp_figure(
            config=config.map,
            cluster_map=results.cluster_map,
            title=title,
            n_comp=config.cluster.n_comp,
        )
        IO.saveFigure(Path(f"{v_prefix}map_comp.png"), map_comp_fig)

    if len(config.map.keywords) == 4:
        # two 2d radiance scatter plots
        plot_fig = PLP.create_plots_figure(
            config=config.map,
            input_arr=input_obj.data,
            cluster_map=results.cluster_map,
            title=title,
            pred=results.pred,
            n_comp=config.cluster.n_comp,
        )
        IO.saveFigure(Path(f"{v_prefix}plot.png"), plot_fig)

    # heat map for mean value for each cluster for each parameter
    centroids_fig = STAT.get_centroids_figure(config.map.keywords, results.means, title)
    centroids_fig.savefig(f"{v_prefix}centroids.png")

    # spatial map of clusters
    map_fig: Figure = PLP.create_cluster_map(
        config=config.map,
        cluster_map=results.cluster_map,
        title=title,
        n_comp=config.cluster.n_comp,
    )
    map_fig.savefig(f"{v_prefix}cluster_map.png")

    # spatial maps for each cluster indicating probability the pixel belongs to that cluster
    uncertainty_fig = PP.create_uncertainty_fig(
        config=config.map,
        prob_map=results.prob_map,
        n_comp=config.cluster.n_comp,
        title=title,
    )
    uncertainty_fig.savefig(f"{v_prefix}uncertainty_map.png")

    # spatial map indicating maximum posterior probability for each pixel
    max_prob_fig = PP.create_max_prob_map(
        config=config.map,
        prob_map=results.prob_map,
        title=title,
    )
    max_prob_fig.savefig(f"{v_prefix}max_prob_fig.png")

    if config.cluster.isPca:
        # gets the loadings of each cluster on each pca
        heat_map_fig = PCA.get_loadings_heatmap(
            results.pca_obj, config.map.keywords, title
        )
        heat_map_fig.savefig(f"{v_prefix}loadings.png")
