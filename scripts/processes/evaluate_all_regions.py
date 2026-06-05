import pandas as pd

import scripts.clustering.cluster_evaluation as EV

# change directory structure to name(from lon)

def get_lat_range(rng):
    lat1, lat2 = rng.split("-")
    dir1, dir2 = lat1[-1], lat2[-1]
    lat_range = []
    lat_range[0] = -1 * lat1[:-1] if dir1 == "S" else lat1[:-1]
    lat_range[1] = -1 * lat1[:-1] if dir2 == "S" else lat1[:-1]
    return lat_range

def run_all_eval_ROI(r):
    if(r["GMM"]):
        return
    lat_range = get_lat_range(r["PG Lat Rng"])
    lng_range = r['Sys 1 Long Rng']
    ROI = r['ROI Dict']
    source = r["Data Source"]
    



regions_data = pd.read_csv("regions.csv")
print(regions_data.columns)

regions_data.apply(run_all_eval_ROI, axis = 1)

