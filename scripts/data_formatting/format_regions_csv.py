
import pandas as pd

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


def format_to_df():
    """
    Returns
    -----------------
    data frame created from "regions.csv" with processed latitude and longitude limits
    """
    regions_data = pd.read_csv("regions.csv")
    rows = []
    for _, r in regions_data.iterrows():
        if(r["GMM"]):
            return
  
        lat_range = get_lat_range(r["PG Lat Rng"])
        lng_range = list(map(int, r['Sys 1 Long Rng'].split("-")))
        lat_lims = [360 - lng_range[1], 360 - lng_range[0]]
        lon_lims = [90 - lat_range[1], 90 - lat_range[0]]
        new_row = r.copy()
        new_row["lon_lims"] = lon_lims
        new_row["lat_lims"] = lat_lims
        rows.append(new_row)
    new_df = pd.DataFrame(rows)
    return new_df

if __name__ == "__main__":
    print(format_to_df())

