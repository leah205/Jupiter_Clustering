import scripts.clustering.BIC as BIC
import scripts.clustering.silhouette as SIL
import scripts.clustering.gmm_distance as JS
import scripts.preprocessing.preprocessing as pre
import scripts.pca as PCA


def raw_evaluation_pipeline(subdir, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           cluster_rng = [2, 10],
                            cm_num = 1):
    
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pix_arr = input_arr[:, 0: len(keywords)] 
  
    print("doing bic:")
    fig_bic  = BIC.create_bic_plot(pix_arr, cluster_rng)
    print("doing sil:")
    fig_sil = SIL.silhouette_graph(pix_arr, cluster_rng)
    print("doing js:")
    fig_js = JS.create_js_plot(pix_arr, cluster_rng)
    keyword_str = "_".join(keywords)

    fig_bic.savefig(f"cluster_evaluations/{subdir}/BIC_plot_{keyword_str}")
    fig_sil.savefig(f"cluster_evaluations/{subdir}/sil_plot_{keyword_str}")
    fig_js.savefig(f"cluster_evaluations/{subdir}/js_plot_{keyword_str}")

def pca_evaluation_pipeline(subdir, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           cluster_rng = [2, 10],
                            cm_num = 1):
    
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pix_arr = input_arr[:, 0: len(keywords)] 
    [pca_reduced, pca_obj] = PCA.get_pca_comp(input_arr[:, 0:len(keywords)])
    keyword_str = "_".join(keywords)

    print("doing bic:")
    fig_bic  = BIC.create_bic_plot(pca_reduced, cluster_rng)
    fig_bic.savefig(f"cluster_evaluations/{subdir}/BIC_plot_{keyword_str}_pca")
    print("doing sil:")
    fig_sil = SIL.silhouette_graph(pca_reduced, cluster_rng)
    fig_sil.savefig(f"cluster_evaluations/{subdir}/sil_plot_{keyword_str}_pca")
    print("doing js:")
    fig_js = JS.create_js_plot(pca_reduced, cluster_rng)
    fig_js.savefig(f"cluster_evaluations/{subdir}/js_plot_{keyword_str}_pca")





    

lon_range = [0, 30]
lat_range = [0, 15]

lon_range = [360 - lon_range[1], 360 - lon_range[0]]
lat_range = [90 - lat_range[1], 90 - lat_range[0]]

cluster_rng = [2, 12]

raw_evaluation_pipeline("20251016UTc/lon_0-30", ["NH3", "PCld"], [[0, 300], [1000,  3200]], lat_range, lon_range, cluster_rng, 1)
# raw_evaluation_pipeline("20251016UTc/lon_0-30", ["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)
# pca_evaluation_pipeline("20251016UTc/lon_0-30", ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
#                  [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], lat_range, lon_range, cluster_rng, 1)


#raw_evaluation_pipeline("20251016UTc/lon_0-30", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)

#pca_evaluation_pipeline("20251016UTc/lon_0-30", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, cluster_rng, 1)

# raw_evaluation_pipeline("20251016UTc/lon_0-30", [ "619", "631", "645"], 
#                  [[0, 1], [0,  1], [0,  1]], lat_range, lon_range, cluster_rng, 1)