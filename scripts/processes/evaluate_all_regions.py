import pandas as pd
import ast
import scripts.clustering.cluster_evaluation as EV
import config.types as T
import config.dicts as D
from scripts.data_formatting.parse_map_to_df import parse_map_to_df


def get_lat_range(rng):
    """
        Parameters
        -----------
        rng: string
            - two numbers followed by S(south) or N(north) separated by -
        Returns
        ------------
        two element list of min and max latitude between -90 and 90
    """
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
    """
        Runs evaluation pipeline on region of interest for each clustering run specified in config
    
        Parameters
        ----------------
        r: row in pd data frame
    
    
        Returns
        ------------
        void
        """
  
  
    ROI = ast.literal_eval(r['ROI Dict'].split("=")[1])
    lat_lims , lon_lims = r["lat_lims"], r["lon_lims"]
    source = r["Data Source"]


    for cl_info in D.eval_runs:
        mapConfig =  T.mappingConfig(
        keywords = cl_info["dims"],
        ROI = ROI,
        latRng = lat_lims,
        lngRng = lon_lims,
        name = r["Name"],
        source = source,
        cm_num = 1
        )
        if(cl_info["PCA"]):
            EV.pca_evaluation_pipeline(mapConfig)
        else:
            EV.raw_evaluation_pipeline(mapConfig)

if __name__ == "__main__":
    regions_data = parse_map_to_df()

    regions_data.apply(run_all_eval_ROI, axis = 1)

