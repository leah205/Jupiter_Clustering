import scripts.clustering.BIC as BIC
import scripts.clustering.silhouette as SIL
import scripts.clustering.gmm_distance as JS
import scripts.preprocessing.preprocessing as pre
import scripts.pca as PCA
import config.types as TY
import config.dicts as D
import scripts.io as IO

from pathlib import Path

def get_evaluation_output_dir(c, m):
    pca_dir = ("PCA/") if c.isPca else ""
    return f'cluster_evaluations/{m.name}/{pca_dir}'

     



def run_evaluation_pipeline(config, cluster_rng=[2,10]):
    """
    Creates BIC, JS, silhouette plots for all listed dimensions and PCA combinations in eval_runs


    Parameters
    ----------
    config: PipelineConfig
    cluster_rng: python list
        - pair of values specifying minimum and maximum clusters numbers to evaluate

    """
    output_dir = get_evaluation_output_dir(config.cluster, config.map)
    output_dir_path = Path(output_dir)
    
    #clustering has already been run
    if(output_dir_path.exists()):
            return

    output_dir_path.mkdir(parents = True, exist_ok = True)

    keywords = config.map.keywords
    input_dir = IO.get_dir_path(config.map)
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]
    fits_files = IO.get_fits_files(input_dir, keywords)
    [pix_arr, subset_shape, indices] = pre.get_input_array(config.map, param_ranges, fits_files)
    input = pix_arr
    if(config.cluster.isPca):
         reduced, obj, scaler = PCA.get_pca_comp(pix_arr)
         input = reduced
    print("doing bic:")
    prefix = output_dir_path 
    keyword_str = '_'.join(config.map.keywords)

    fig_bic  = BIC.create_bic_plot(pix_arr, cluster_rng)
    fig_bic.savefig(f"{prefix}/BIC_plot_{keyword_str}.png")
    print("doing sil:")
    fig_sil = SIL.silhouette_graph(pix_arr, cluster_rng)
    fig_sil.savefig(f"{prefix}/sil_plot_{keyword_str}.png")
    print("doing js:")
    fig_js = JS.create_js_plot(pix_arr, cluster_rng)
    fig_js.savefig(f"{prefix}/js_plot_{keyword_str}.png")
    



