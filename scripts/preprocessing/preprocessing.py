from config.config import config
from os import listdir
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS

def get_radiance_arr(file):
    hdul = fits.open(file)
    return hdul[0].data

def get_wcs(file):
    hdul = fits.open(file)
    hdr = hdul[0].header
    return WCS(hdr)

def is_files_aligned(file_arr):
    ref_wcs = get_wcs(file_arr[0])
    for file in file_arr:
        if(ref_wcs.wcs.compare(get_wcs(file).wcs) == False):
            return False
    return True



def get_file_path(keyword, dir):
    for f in listdir(dir):
        if keyword in f:
            return dir + "/" + f
           
def get_parameter_2d_array(keyword_arr, latLims, lonRng):
    '''
    Builds parameter array within lon/lat range of pixel radiances for specified keywords

    Parameters
    -----------
    keyword_arr, MANDATORY
        Description: array of keywords to select files
    latLims, MANDATORY
        Description: Two element array specifying min and max latitudes
    lonRng, MANDATORY
        Description: Two element array specifying min and max longitudes
    
    Returns
    ----------
    2D numpy array with parameters on axis 0 and pixels on axis 1 

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

def get_map_shape(keyword, latLims, longLims):
      dir_path = config["input"]
      file = get_file_path(keyword, dir_path)
      radiance_arr = get_radiance_arr(file)
      return subset_map(radiance_arr, latLims, longLims).shape

def get_filtered_pix_arr(range_arr, pixel_arr):
  
    '''
    
    Parameters
    -----------
    range_arr: Python array with subarrays of two elements: [min, max]
    pixel_arr:  2D numpy array with each row being a pixel and each parameter being a column
    Returns
    ----------
    filtered pix_arr with only pixel rows with all parameters in specified range
    '''

    range_arr = np.array(range_arr)
 
    mins = range_arr[:, 0]
    maxs = range_arr[:, 1]
    #don't want to filter on last column(index)
    #want to filter up to pixel_arr.shape - 2 
    pixel_param_cols = pixel_arr[:, 0:pixel_arr.shape[1] - 1]
    mask = (pixel_param_cols >= mins) & (pixel_param_cols <= maxs) 
    filtered = pixel_arr[np.all(mask, axis = 1)]
   
    return filtered

def get_mapped_pix_arr(pix_arr):
    indices = np.arange(pix_arr.shape[0])
    mapped_pix_arr =np.insert(pix_arr, pix_arr.shape[1], indices, axis = 1)
    return mapped_pix_arr


def get_pix_arr(param_arr, range_arr = []):
    '''
    Main preprocessing routine that returns array to be passed into clustering

    Parameters 
    ----------
    range_arr: Optional
        Description: 2D array of parameter ranges
        Default: Empty Array
    keywords: Optional
        Description: Array of keywords specifying file for pixel radiance
        Default: ["PCld", "NH3"]
    latLims: Optional
        Description: Two element array specifying min and max latitudes
        Default: [45, 135]
    lonLims: Optional
        Description: Two element array specifying min and max longitudes
        Default: [0, 360]
    
    Returns 
    --------
    numpy array with axis 0 as pixels within lon/lat range and axis 1 as parameter pixel radiances and index,
    filtered with rangeArr
    '''
    #if(len(range_arr) and len(range_arr) != len(keywords)):
     #   raise TypeError("ranges array should match number of parameters")
    print(range_arr)
    pixel_arr = np.column_stack(param_arr)
    mapped_pixel_arr = get_mapped_pix_arr(pixel_arr)
    if(len(range_arr)):
        filtered_arr = get_filtered_pix_arr(range_arr, mapped_pixel_arr)
        return filtered_arr
    return pixel_arr

    
def subset_map(map, latLims, lonRng):
    scale = int(map.shape[0]/ 180)
    latLims = np.array(latLims)* scale
    lonRng= lonRng *scale
    #print(latLims)
    #print(lonRng)
    #print(map[latLims[0]:latLims[1], lonRng[0]: lonRng[1]])
    return map[latLims[0]:latLims[1], lonRng[0]: lonRng[1]]

#get_pix_arr([[60, 160], [1400, 2200]], ["NH3", "PCld"])

