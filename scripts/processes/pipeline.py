from pathlib import Path

import config.dicts as D
import config.types as TY
import scripts.clustering.clusters as CL
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

    print("yoohoo")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    keywords = config.map.keywords
    input_dir = IO.get_dir_path(config.map)
    fits_files = IO.get_fits_files(input_dir, keywords)
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]

    # function to cluster
    transform = CL.run_pca_pipeline if config.cluster.isPca else CL.run_raw_pipeline

    # get input numpy array and shape filtered by coordinates and radiance values
    [data, subset_shape, indices] = pre.get_input_array(
        config.map, param_ranges, fits_files
    )

    # run clustering function on processed data
    cluster_obj = transform(data, config.cluster)
    pred, probs, means, covariances = (
        cluster_obj["pred"],
        cluster_obj["probs"],
        cluster_obj["means"],
        cluster_obj["covariances"],
    )

    # map cluster outputs to original map shape (within coordinate range)
    reshaped_pred = MP.reshape_clustered(indices, subset_shape, pred)

    prefix = output_dir_path / IO.create_file_prefix(
        config.cluster, config.map, fits_files[0]
    )
    title = PLP.create_plot_title(config.cluster, config.map)

    if len(keywords) == 2:
        # two-d radiance scatter plot
        plot_fig = PLP.create_plot_figure(
            config=config.map,
            arr=data,
            reshaped_pred=reshaped_pred,
            title=title,
            cluster_obj=cluster_obj,
            n_comp=config.cluster.n_comp,
        )
        IO.saveFigure(Path(f"{prefix}plot.png"), plot_fig)
        # comparison of radiances in different filters mapped spatially
        map_comp_fig = PLP.create_map_comp_figure(
            config=config.map,
            reshaped_pred=reshaped_pred,
            title=title,
            n_comp=config.cluster.n_comp,
        )
        IO.saveFigure(Path(f"{prefix}map_comp.png"), map_comp_fig)

    if len(keywords) == 4:
        # two 2d radiance scatter plots
        plot_fig = PLP.create_plots_figure(
            config=config.map,
            input_arr=data,
            reshaped_pred=reshaped_pred,
            title=title,
            pred=pred,
            n_comp=config.cluster.n_comp,
        )
        IO.saveFigure(Path(f"{prefix}plot.png"), plot_fig)

    # heat map for mean value for each cluster for each parameter
    centroids_fig = STAT.get_centroids_figure(keywords, means, title)
    centroids_fig.savefig(f"{prefix}centroids.png")

    # spatial map of clusters
    map_fig = PLP.create_cluster_map(
        config=config.map,
        reshaped_pred=reshaped_pred,
        title=title,
        n_comp=config.cluster.n_comp,
    )
    map_fig.savefig(f"{prefix}cluster_map.png")

    stats = STAT.get_all_stats(
        cluster_obj=cluster_obj,
        indices=indices,
        n_comp=config.cluster.n_comp,
        map_config=config.map,
        data=data,
    )
    # dump means and standard deviations per cluster per parameter into json
    STAT_IO.save_stats_json(config, stats)

    # spatial maps for each cluster indicating probability the pixel belongs to that cluster
    uncertainty_fig = PP.create_uncertainty_fig(
        config=config.map,
        probs=probs,
        indices=indices,
        subset_shape=subset_shape,
        title=title,
    )
    uncertainty_fig.savefig(f"{prefix}uncertainty_map.png")

    # spatial map indicating maximum posterior probability for each pixel
    max_prob_fig = PP.create_max_prob_map(
        config=config.map,
        probs=probs,
        indices=indices,
        subset_shape=subset_shape,
        title=title,
    )
    max_prob_fig.savefig(f"{prefix}max_prob_fig.png")

    if config.cluster.isPca:
        # gets the loadings of each cluster on each pca
        heat_map_fig = PCA.get_loadings_heatmap(cluster_obj["pca_obj"], keywords, title)
        heat_map_fig.savefig(f"{prefix}loadings.png")
