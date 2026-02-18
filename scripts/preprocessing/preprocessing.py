from config.config import config
from os import listdir
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
#create errors
def get_radiance_arr(file):
    hdul = fits.open(file)
    
    #print(hdul[0].data)
    return hdul[0].data

def get_wcs(file):
    print(file)
    hdul = fits.open(file)
    hdr = hdul[0].header
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
           
def get_parameter_nd_array(keyword_arr):
    
    dir_path = config["input"]
    file_arr = []
    file_name_arr = []
    for keyword in keyword_arr:
        file = get_file_path(keyword, dir_path)
        file_arr.append(get_radiance_arr(file))
        file_name_arr.append(file)


    #nh3_file = get_file_path("NH3", dir_path)
    #cld_file = get_file_path("Cld", dir_path)
    

    res = np.array(file_arr)
    if(is_files_aligned(np.array(file_name_arr)) == False):
        raise TypeError("files are not mapped to the same coordinates")
    print(res.ndim)
    print(res.shape)
    return res




print(type(get_parameter_nd_array(["NH3", "PCld"])))