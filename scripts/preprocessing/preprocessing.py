from config.config import config
from os import listdir
from astropy.io import fits
#from reproject import reproject_interp
import numpy as np
from astropy.wcs import WCS

def get_radiance_arr(file):
    hdul = fits.open(file)
    return hdul[0].data

def get_wcs(file):
    hdul = fits.open(file)
    hdr = hdul[0].header
    return WCS(hdr)

def get_header_key(file, key):
    hdul = fits.open(file)
    hdr = hdul[0].header
    return hdr[key]


def get_cm(first_file, cm_num):
    ctype_key = {
         1:'CM1',
         2:'CM2',
         3:'CM3',
    }
    return get_header_key(first_file, cm_num)


def is_files_aligned(file_arr):
    ref_wcs = get_wcs(file_arr[0])
    
    for file in file_arr:
        
        if(ref_wcs.wcs.compare(get_wcs(file).wcs) == False):
            return False
    return True
'''
def get_radiances(file_arr):
    #ref_file = fits.open(file_arr[0])
    ref_wcs = get_wcs(file_arr[0])
    ref_hdr = fits.open(file_arr[0])[0].header
    new_arr = []
    for file in file_arr:
        
        if(ref_wcs.wcs.compare(get_wcs(file).wcs) == False):
            hdul = fits.open(file)
            radiance_arr = reproject_interp(hdul, ref_hdr)[0]
        else:
            radiance_arr = get_radiance_arr(file)
'''

def get_radiances(file_arr):
    radiances = []
    for file in file_arr:
       radiances.append(get_radiance_arr(file))
    return radiances



def get_file_path(keyword, dir):
    for f in listdir(dir):
        if keyword in f:
            return dir + "/" + f
           
def get_parameter_2d_array(keyword_arr):
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
    file_name_arr = []
    for keyword in keyword_arr:
        file = get_file_path(keyword, dir_path)
        file_name_arr.append(file)

    ''' radiances_arr = []
    if(is_files_aligned(np.array(file_name_arr)) == False):
        get_align_files(np.array(file_name_arr)
        raise TypeError("files are not mapped to the same coordinates")

    for file_name in file_name_arr:
        radiance_arr = get_radiance_arr(file_name)
        radiances_arr.append(radiance_arr)

    return radiances_arr'''
    radiances = get_radiances(file_name_arr)
    return radiances

def get_map_shape(keyword, latLims, longLims):
      dir_path = config["input"]
      file = get_file_path(keyword, dir_path)
      radiance_arr = get_radiance_arr(file)
      return subset_map(radiance_arr, latLims, longLims).shape


def get_patch(keyword, latLims, lngLims, cm_num = 3):
    

    file = get_file_path(keyword, config["input"])
    CM = 0 if cm_num == 0 else get_cm(file, cm_num)
    radiance_arr = get_radiance_arr(file)
    return subset_map(radiance_arr, latLims, lngLims, CM)


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
    if(len(range_arr) and len(range_arr) != pixel_arr.shape[1] - 1):
        raise TypeError("ranges array should match number of parameters")
    

    range_arr = np.array(range_arr)
    mins = range_arr[:, 0]
    maxs = range_arr[:, 1]
    #don't want to filter on last column(index)
    pixel_param_cols = pixel_arr[:, 0:range_arr.shape[0]]
    mask = (pixel_param_cols >= mins) & (pixel_param_cols <= maxs) 
    filtered = pixel_arr[np.all(mask, axis = 1)]
   
    return filtered

def get_mapped_pix_arr(pix_arr):
    '''
    Parameters
    ----------

    pix_arr, TYPE NONOPTIONAL
    DESCRIPTION, 2d numpy array with pixels as rows (axis 0) and parameters as columns ( axis 1 )

    Returns
    ----------

    returns 2d numpy array with column added for indices
    '''
    indices = np.arange(pix_arr.shape[0])
    mapped_pix_arr =np.insert(pix_arr, pix_arr.shape[1], indices, axis = 1)
    return mapped_pix_arr
 
    
def subset_map(map, LatLims, LonLims,  CM):
    import numpy as np
    import copy

    LonRng = (LonLims[1] - LonLims[0]) / 2
    print("map shape: " + str(map.shape))
    scale=int(map.shape[0]/180)
    print("######## scale=",scale)
    lon_max=360*scale
    LatLims=np.array(LatLims)*scale
    LonRng=LonRng*scale
    CM=CM*scale
    LonLims=np.array(LonLims)*scale
    
    print(lon_max,LatLims,LonRng,CM,LonLims)
    if(CM == 0):
        return np.copy(map[LatLims[0]:LatLims[1],LonLims[0]:LonLims[1]])
   
    
    if CM<LonRng:
        #crosses longitude boundary to the left of CM
        print("******************  CM2deg<LonRng")
        #slices of higher longitudes, lower longitudes concatenated
        patch=np.concatenate((np.copy(map[LatLims[0]:LatLims[1],LonLims[0]-1:lon_max]),
                              np.copy(map[LatLims[0]:LatLims[1],0:LonLims[1]-lon_max])),axis=1)
    if CM>lon_max-LonRng:
        #crossses longitude boundary to the right of CM
        print("******************  CM2deg>LonRng")
        #slices of higher longitudes, lower longitudes concatenated
        patch=np.concatenate((np.copy(map[LatLims[0]:LatLims[1],lon_max+LonLims[0]:lon_max]),
                              np.copy(map[LatLims[0]:LatLims[1],0:LonLims[1]])),axis=1)
        print("lon_max+LonLims[0]:lon_max,0:LonLims[1]=",lon_max+LonLims[0],lon_max,0,LonLims[1])
    print("patch shape:" + str(patch.shape))
    return patch    



def get_input_array(keywords, param_ranges,
                        latRng, lngRng, cm_num):
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
    radiances_arr = get_parameter_2d_array(keywords)

    subpatches = []
    first_file = get_file_path(keywords[0], config["input"])

    CM = 0 if cm_num == 0 else get_cm(first_file, cm_num)
    subset_shape = (0,0)

    for radiance_arr in radiances_arr:
        subset = subset_map(radiance_arr, latRng, lngRng, CM)
        subset_shape = subset.shape
        subpatches.append(subset.flatten())

    subpatches = np.array(subpatches)
    pix_arr = np.column_stack(subpatches)
    pix_arr = get_mapped_pix_arr(pix_arr)
    pix_arr = get_filtered_pix_arr(param_ranges, pix_arr)
  
    return [pix_arr, subset_shape]
    



