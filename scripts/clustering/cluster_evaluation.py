import scripts.clustering.BIC as BIC
import scripts.clustering.silhouette as SIL
import scripts.clustering.gmm_distance as JS
import scripts.preprocessing.preprocessing as pre
import scripts.pca as PCA
import config.types as TY
import config.dicts as D
from pathlib import Path


def raw_evaluation_pipeline(mapConfig, 
                           cluster_rng = [2, 10],
                            ):
    """
    Parameters
    ---------------
    mapConfig
    cluster_rng: int[]
        - range of clustersto evaluate


    Saves into cluster_evaluations directory graphs of BIC, silhouette score, and JS distance
    """
  
    keywords = mapConfig.keywords
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]
    [input_arr, subset_shape] = pre.get_input_array(mapConfig, param_ranges)
    pix_arr = input_arr[:, 0: len(keywords)] 
  
    
    keyword_str = "_".join(keywords)
    filepath_str = f"cluster_evaluations/{mapConfig.source}/{mapConfig.name}/{keyword_str}"
    filepath = Path(filepath_str)
    filepath.mkdir(parents=True, exist_ok=True)
  
    print("doing bic:")
    fig_bic  = BIC.create_bic_plot(pix_arr, cluster_rng)
    fig_bic.savefig(f"{filepath_str}/BIC_plot_{keyword_str}")
    print("doing sil:")
    fig_sil = SIL.silhouette_graph(pix_arr, cluster_rng)
    fig_sil.savefig(f"{filepath_str}/sil_plot_{keyword_str}")
    print("doing js:")
    fig_js = JS.create_js_plot(pix_arr, cluster_rng)
    fig_js.savefig(f"{filepath_str}/js_plot_{keyword_str}")

def pca_evaluation_pipeline(mapConfig, 
                           cluster_rng = [2, 10],
                            ):
    """
        Parameters
        ---------------
        mapConfig
        cluster_rng: int[]
            - range of clustersto evaluate
    
    
        Saves into cluster_evaluations directory graphs of BIC, silhouette score, and JS distance for PCA-reduced analysis
        """
      
    keywords = mapConfig.keywords
    param_ranges = [D.ranges_dict.get(keyword, [0, 1]) for keyword in keywords]
    [input_arr, subset_shape] = pre.get_input_array(mapConfig, param_ranges)
    pix_arr = input_arr[:, 0: len(keywords)] 
    reduced, obj, scaler = PCA.get_pca_comp(input_arr[:, 0:len(keywords)])
    
    keyword_str = "_".join(keywords)
    filepath_str = f"cluster_evaluations/{mapConfig.source}/{mapConfig.name}/{keyword_str}"
    filepath = Path(filepath_str)
    filepath.mkdir(parents=True, exist_ok=True)



    print("doing bic:")
    fig_bic  = BIC.create_bic_plot(reduced, cluster_rng)
    fig_bic.savefig(f"{filepath_str}/BIC_plot_{keyword_str}")
    print("doing sil:")
    fig_sil = SIL.silhouette_graph(reduced, cluster_rng)
    fig_sil.savefig(f"{filepath_str}/sil_plot_{keyword_str}")
    print("doing js:")
    fig_js = JS.create_js_plot(reduced, cluster_rng)
    fig_js.savefig(f"{filepath_str}/js_plot_{keyword_str}")





# # region = R.ROI_3

# lon_range = region["lon_range"]
# lat_range = region["lat_range"]
# ROI = region["ROI"]

# lon_str = f"lon_{lon_range[0]}-{lon_range[1]}"

# lon_range = [360 - lon_range[1], 360 - lon_range[0]]
# lat_range = [90 - lat_range[1], 90 - lat_range[0]]

# cluster_rng = [2, 12]


# exp1Config = TY.mappingConfig(
#         keywords = ["275", "395", "502", "619", "631", "645", "673", "727", "889"],
#         ROI = ROI,
#         latRng = lat_range,
#         lngRng = lon_range
#         )

if __name__ == "__main__":
    print("empty")
    # pca_evaluation_pipeline(f"20251016UTc/{lon_str}", exp1Config, cluster_rng)
    # raw_evaluation_pipeline("20251016UTc/lon_0-30", ["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)
    # pca_evaluation_pipeline("20251016UTc/lon_0-30", ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
    #                  [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], lat_range, lon_range, cluster_rng, 1)


    #raw_evaluation_pipeline("20251016UTc/lon_0-30", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)

    #pca_evaluation_pipeline("20251016UTc/lon_0-30", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)

    # raw_evaluation_pipeline("20251016UTc/lon_0-30", [ "619", "631", "645"], 
    #                  [[0, 1], [0,  1], [0,  1]], lat_range, lon_range, cluster_rng, 1)