from config.config import config
from os import listdir
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
#add dimension error
#add get_pix_arr test
#regression tests?
def get_radiance_arr(file):
    hdul = fits.open(file)
    return hdul[0].data

def get_wcs(file):
    print(file)
    hdul = fits.open(file)
    hdr = hdul[0].header
    print(WCS(hdr))
    return WCS(hdr)

def is_files_aligned(file_arr):
    #(w1.wcs.compare(w2.wcs))
    ref_wcs = get_wcs(file_arr[0])
    print(ref_wcs)
    for file in file_arr:
        if(ref_wcs.wcs.compare(get_wcs(file).wcs) == False):
            return False
    return True



def get_file_path(keyword, dir):
    for f in listdir(dir):
        if keyword in f:
            return dir + "/" + f
           
def get_parameter_2d_array(keyword_arr, latLims = [45, 135], lonRng = [0, 360]):
    '''
    Purpose:


    Parameters
    ------------
    '''
   
    dir_path = config["input"]
    file_arr = []
    file_name_arr = []
    for keyword in keyword_arr:
        file = get_file_path(keyword, dir_path)
        file_name_arr.append(file)

    if(is_files_aligned(np.array(file_name_arr)) == False):
        raise TypeError("files are not mapped to the same coordinates")

    for file_name in file_name_arr:
        radiance_arr = get_radiance_arr(file_name)
        file_arr.append(subset_map(radiance_arr, latLims, lonRng).flatten())
    res = np.array(file_arr)
    return res

def get_pix_arr(arr):
    parameter_arr = get_parameter_2d_array(arr)
    return np.column_stack((parameter_arr))

def subset_map(map, latLims, lonRng):
    '''
    Purpose:


    Parameters
    ------------
    '''
    scale = int(map.shape[0]/ 180)
    latLims = np.array(latLims)* scale
    lonRng=lonRng*scale
    print(latLims, lonRng)
    return map[latLims[0]:latLims[1], lonRng[0]: lonRng[1]]



print(type(get_parameter_2d_array(["NH3", "PCld"])))