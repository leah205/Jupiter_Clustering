import pandas as pd
import ast
import scripts.clustering.cluster_evaluation as EV
import config.types as T
import config.dicts as D
import scripts.pipeline as PIPE

def get_lat_range(rng):
    lat1, lat2 = rng.split("-")
    dir1, dir2 = lat1[-1], lat2[-1]
    if(not dir1.isnumeric()):
        lat1 = lat1[:-1]
    if(not dir2.isnumeric()):
        lat2 = lat2[:-1]
    lat1, lat2 = int(lat1), int(lat2) 
    lat1 = -1 * lat1 if dir1 == "S" else lat1
    lat2 = -1 * lat2 if dir2 == "S" else lat2
    return [lat1, lat2]



def run_all_eval_ROI(r):
    if(r["GMM"]):
        return
  
    lat_range = get_lat_range(r["PG Lat Rng"])
    lng_range = list(map(int, r['Sys 1 Long Rng'].split("-")))
    lng_range = [360 - lng_range[1], 360 - lng_range[0]]
    lat_range = [90 - lat_range[1], 90 - lat_range[0]]
    ROI = ast.literal_eval(r['ROI Dict'].split("=")[1])
    source = r["Data Source"]

    

    for cl_info in D.cluster_runs:
        mapConfig =  T.mappingConfig(
        keywords = cl_info["dims"],
        ROI = ROI,
        latRng = lat_range,
        lngRng = lng_range,
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
        
        

       
    
regions_data = pd.read_csv("regions.csv")

regions_data.apply(run_all_eval_ROI, axis = 1)
