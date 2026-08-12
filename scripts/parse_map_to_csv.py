from hst_study_maps import Study_Maps
import pandas as pd

def get_lon_lims(cmpref, lon_rng):
    return [cmpref - lon_rng / 2, cmpref + lon_rng / 2]
def get_lat_lims(lat_lims):
    return [lat_lims[0] - 90, lat_lims[1] - 90]

def parse_map_to_csv():
    rows = []
    for obs, value in Study_Maps.items():
        sys1 = value['1']
        for key, val in sys1.items():
            if key == 'CoLatLims':
                # lat_rng = sys1['CoLatLims']
                continue
            elif key == 'LonRng':
                # lon_rng = sys1['LonRng']
                continue
            elif key == 'CMpref':
                # center_lon = sys1['CMpref']
                continue
            elif key == 'plotoptions':
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
                    "Source": obs,
                    "lat_lims": lat_lims,
                    "lon_lims": lon_lims,
                    "ROI Dict": roi_dict
                }
                rows.append(row)


    df = pd.DataFrame(rows)
    df.to_csv("new_regions.csv", index = False)

if __name__ == "__main__":
    parse_map_to_csv()
    



