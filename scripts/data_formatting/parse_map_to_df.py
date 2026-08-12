from hst_study_maps import Study_Maps
import pandas as pd

def get_lon_lims(cmpref, lon_rng):
    """
    Parameters
    ------------
    cmpref: float
        - indicates center of longitude range
    lon_rng: float
        - indicates width of longitude
    Returns
    -------------
    list of format [min_lon, max_lon] where longitude ranges between 0 - 360

    """
    return [cmpref - lon_rng / 2, cmpref + lon_rng / 2]
def get_lat_lims(lat_lims):
    """
    Parameters
    ---------------
    lat_lims: float[]
        - list of form [min_lat, max_lat], where latitude is between 0 and 180
    Returns
    ---------------
        - float[] of form [min_lat, max_lat], where latitude is between -90 and 90

    
    """
    return [lat_lims[0] - 90, lat_lims[1] - 90]

def parse_map_to_df():
    """
    Parses study map into df that can be input into pipeline

    Parameters
    -----------
    None

    Returns
    ---------
    Pandas df with a row for each region and columns for source, lat_lims, lon_lims, and ROI coordinates
    """
    rows = []
    for obs, value in Study_Maps.items():
        sys1 = value['1']
        for key, val in sys1.items():
            if key in ['CoLatLims', 'LonRng', 'plotoptions', 'CMpref']:
                continue
            else: 
                name = obs + "-" + key
                roi_dict = "ROI=" + str(val['ROI'])
                lon_rng = val['LonRng']
                lat_lims = get_lat_lims(val['CoLatLims'])
                center = val['CMpref']
                lon_lims = get_lon_lims(center, lon_rng)
                row = {
                    "Name": name,
                    "Data Source": obs,
                    "lat_lims": lat_lims,
                    "lon_lims": lon_lims,
                    "ROI Dict": roi_dict
                }
                rows.append(row)


    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    parse_map_to_df()
    



