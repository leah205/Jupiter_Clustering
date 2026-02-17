from config.config import config
from os import listdir
from astropy.io import fits
import numpy as np
#create errors
def get_radiance_arr(file):
    hdul = fits.open(file)
    
    #print(hdul[0].data)
    return hdul[0].data

def get_file_path(keyword, dir):
    for f in listdir(dir):
        if keyword in f:
            return dir + "/" + f
           
def get_parameter_nd_array(keyword_arr):
    dir_path = config["input"]
    file_arr = []
    for keyword in keyword_arr:
        file = get_file_path(keyword, dir_path)
        print(get_radiance_arr(file).shape)
        file_arr.append(get_radiance_arr(file))


    #nh3_file = get_file_path("NH3", dir_path)
    #cld_file = get_file_path("Cld", dir_path)
   
    res = np.array(file_arr)
    print(res.ndim)
    print(res.shape)
    return res




print(type(get_parameter_nd_array(["NH3"])))