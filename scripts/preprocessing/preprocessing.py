
from os import listdir
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import time
from config.config import cf
from scripts.helpers import get_dir_path

def get_radiance_arr(file):
    """
    Parameters
    -----------
    file: string
        - file path
    Returns
    -----------
    numpy array with radiance values for pixels
    """
    hdul = fits.open(file)
    
    return hdul[0].data

def get_wcs(file):
    hdul = fits.open(file)
    hdr = hdul[0].header
    return WCS(hdr)

def get_header_key(file, key):
    """
    Parameters
    ---------
    file: string
        - path to fits file
    key: string
        - key in fits header
    Returns
    ---------
    Value of key in header
    
    """
    hdul = fits.open(file)
    hdr = hdul[0].header
    return hdr[key]


def get_cm(first_file, cm_num):
    """
    first_file: string
        - path to first file in observation
    cm_num: 1 | 2 | 3
        - number specifying coordinate system 
    Returns
    --------
    central meridian coordiantes

    
    """
    return get_header_key(first_file, cm_num)


def is_files_aligned(file_arr):
    """
    Checks that every file in observation is aligned according to world coordinate system

    Parameters
    ------------
    file_arr: string[]
        - list of file paths for files in observation
    Returns
    ------------
    Boolean indicating alignment
    
    """
    ref_wcs = get_wcs(file_arr[0])
    
    for file in file_arr:
        
        if(ref_wcs.wcs.compare(get_wcs(file).wcs) == False):
            return False
    return True


def get_radiances(file_arr):
    """
    Parameters
    ----------
    file_arr: string[]
            - list of file paths for files in observation
    Returns
    ----------
    List of numpy arrays for radiances of each filter image
    """
    radiances = []
    for file in file_arr:
       radiances.append(get_radiance_arr(file))
    return radiances



def get_file_path(keyword, dir):
    """
    Parameters
    -----------
    keyword: string
        - keyword to select file by
    dir: string
        - path to directory to search for file in
    Returns
    ----------
    string of path to file
    """
    for f in listdir(dir):
        if keyword in f and ".fits" in f:
            return dir + "/" + f
           
def get_parameter_2d_array(keyword_arr, dir_path):
    '''
    Builds parameter array within lon/lat range of pixel radiances for specified keywords

    Parameters
    -----------
    keyword_arr, MANDATORY
        Description: array of keywords to select files
    dir_path: string
        - path to directory to get radiances from

    Returns
    ----------
    List of radiance numpy arrays for each keyword
    '''
   
    file_name_arr = []
    for keyword in keyword_arr:
        file = get_file_path(keyword, dir_path) 
        file_name_arr.append(file)
    radiances = get_radiances(file_name_arr)
    return radiances




def get_patch(keyword, latLims, lngLims, cm_num, dir_path):
    """
    keyword: string
    latLims: [min, max]
        - latitude range
    lngLims: [min, max]
        - longitude range
    cm_num: int
        - coordinate system
    dir_path:
        - path to directory with target files

    Returns
    -------------
    np array with pixels filtered by longitude/latitude range
    """
    file = get_file_path(keyword, dir_path)
    CM = 0 if cm_num == 0 else get_cm(file, cm_num)
    radiance_arr = get_radiance_arr(file)
    return subset_map(radiance_arr, latLims, lngLims)


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
 
    
def subset_map(map, LatLims, LonLims):
    """
    map: np array
        - array to filter coordinates
    latLims: [min, max]
            - latitude range
    lngLims: [min, max]
            - longitude range

    Returns
    ------------
    np array
    """
    import numpy as np
    import copy
    
    
    LonRng = (LonLims[1] - LonLims[0]) / 2
    CM = LonLims[0] + LonRng
    #print("map shape: " + str(map.shape))
    scale=int(map.shape[0]/180)
    #print("######## scale=",scale)
    lon_max=360*scale
    LatLims=np.array(LatLims)*scale
    LonRng=LonRng*scale
    # CM=CM*scale
    CM = 50 * scale
    LonLims=np.array(LonLims)*scale
  
    print(lon_max,LatLims,LonRng,CM,LonLims)
    if(CM == 0):
        return np.copy(map[LatLims[0]:LatLims[1],LonLims[0]:LonLims[1]])
    if CM >= LonRng and CM <= lon_max - LonRng:
        patch = np.copy(
            map[
                int(LatLims[0]):int(LatLims[1]),
                int(LonLims[0]):int(LonLims[1])
            ]
        )
   
    elif CM<LonRng:
        #crosses longitude boundary to the left of CM
        #print("******************  CM2deg<LonRng")
        #slices of higher longitudes, lower longitudes concatenated
        #patch=np.concatenate((np.copy(map[LatLims[0]:LatLims[1],LonLims[0]-1:lon_max]),
        #                      np.copy(map[LatLims[0]:LatLims[1],0:LonLims[1]-lon_max])),axis=1)
        
        
        patch=np.concatenate((np.copy(map[LatLims[0]:LatLims[1],LonLims[0]:lon_max]),
                              np.copy(map[LatLims[0]:LatLims[1],0:LonLims[1]-lon_max])),axis=1)
    elif CM>lon_max-LonRng:
        #crossses longitude boundary to the right of CM
        #print("******************  CM2deg>LonRng")
        #slices of higher longitudes, lower longitudes concatenated
        patch=np.concatenate((np.copy(map[LatLims[0]:LatLims[1],lon_max+LonLims[0]:lon_max]),
                              np.copy(map[LatLims[0]:LatLims[1],0:LonLims[1]])),axis=1)
        #print("lon_max+LonLims[0]:lon_max,0:LonLims[1]=",lon_max+LonLims[0],lon_max,0,LonLims[1])
    print("patch shape:" + str(patch.shape))

    return patch    

def get_date(keywords, dir_name):
    """
    dates an observation by the date of the first image taken

    Parameters
    -----------
    keywords: string
        - list of keywords in observation
    dir_name: string
        - path to directory

    Returns
    ----------
    Date firstimage in observation was taken
    
    """
    seconds_past = 0
    first_file = file = get_file_path(keywords[0], dir_name)
    hdr = fits.open(file)[0].header
    first_date_str = hdr["DATE-OBS"][11:19]
    return hdr["DATE-OBS"][0:19]
    ft = time.strptime(first_date_str, "%H:%M:%S")
    for key in keywords:
        file = get_file_path(key, config["input"])
        hdr = fits.open(file)[0].header
        stime = hdr["DATE-OBS"][11:19]
        t = time.strptime(stime, "%H:%M:%S")
        seconds_past = seconds_past + int(ft.time() - t/time())
    print(ft + seconds_past / len(keywords))
    return ft + seconds_past / len(keywords)



def get_input_array(config, param_ranges,
                        ):
    '''
    Main preprocessing routine that returns array to be passed into clustering

    Parameters 
    ----------
    param_ranges: list
        - list of lists specifying minimum and maximum values to be included in the analysis for each keyword
    config: 
        - object containing mapping information for the input array
    
    Returns 
    --------
    numpy array with axis 0 as pixels within lon/lat range and axis 1 as parameter pixel radiances and index,
    filtered with rangeArr
    '''
    dir_path =  get_dir_path(config)
    radiances_arr = get_parameter_2d_array(config.keywords, dir_path)
  

    subpatches = []
    first_file = get_file_path(config.keywords[0], dir_path)
    

    CM = 0 if config.cm_num == 0 else get_cm(first_file, config.cm_num)
    subset_shape = (0,0)
    
    for radiance_arr in radiances_arr:
        subset = subset_map(radiance_arr, config.latRng, config.lngRng)
        subset[subset == 0] = np.nan
        subset_shape = subset.shape
        subpatches.append(subset.flatten())

    subpatches = np.array(subpatches)
    pix_arr = np.column_stack(subpatches)
    pix_arr = get_mapped_pix_arr(pix_arr)
    pix_arr = get_filtered_pix_arr(param_ranges, pix_arr)
  
    return [pix_arr, subset_shape]
    



