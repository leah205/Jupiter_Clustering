import pandas as pd
import ast
import scripts.clustering.cluster_evaluation as EV
import config.types as T
import config.dicts as D
import scripts.pipeline as PIPE
from scripts.data_formatting.parse_map_to_df import parse_map_to_df


def run_all_cluster_ROI(r):
    """
    Runs clustering pipeline on region of interest for each clustering run specified in config

    Parameters
    ----------------
    r: row in pd data frame


    Returns
    ------------
    void
    """

    ROI = ast.literal_eval(r['ROI Dict'].split("=")[1])
    source = r["Data Source"]
    if(not "202512" in r["Name"]):
        return
    print(source, r["Name"])

   
    lat_lims , lon_lims = r["lat_lims"], r["lon_lims"]
    print(lat_lims, lon_lims)
    
    for cl_info in D.cluster_runs:
        mapConfig =  T.mappingConfig(
        keywords = cl_info["dims"],
        ROI = ROI,
        latRng = lat_lims,
        lngRng = lon_lims,
        name = r["Name"],
        source = source,
        cm_num = 1
        )
     
     
        for n_comp in cl_info["comps"]:
            clusterConfig = T.clusterConfig(
                n_comp=n_comp,
                isPca = cl_info["PCA"]
            )
            config = T.pipelineConfig(
                map = mapConfig,
                cluster = clusterConfig
            )
            PIPE.run_full_pipeline(config)
        
        

if __name__ == "__main__":
    df = parse_map_to_df()
    df.apply(run_all_cluster_ROI, axis=1)
   

    
