import os as OS
import config.types as T
import scripts.preprocessing.preprocessing as pre
from pathlib import Path
from config.config import cf



def get_dir_path(map):
    # return f"HST GO 18055/{source[:10]}/{source}/L3/"
    return f"./data/HST_new/{map.source[:10]}/{map.source}/Sys{map.cm_num}/"


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
    for f in OS.listdir(dir):
        if keyword in f and ".fits" in f:
            return dir + "/" + f

def get_fits_files(dir, keywords):
    files = []
    for keyword in keywords:
        file = get_file_path(keyword, dir)
        files.append(file)
    return files

def saveFigure(path, figure):
    # if not path.exists():
    figure.savefig(path)

def get_output_dir(c: T.clusterConfig, m: T.mappingConfig):
   
    keyword_str = '_'.join(m.keywords)
    pca_dir = ("PCA/") if c.isPca else ""
    thresh_dir = (f"{c.threshold_type}_{c.threshold}/") if cf["soft_clustering"] else ""
    return f'{cf["output"]}/{m.name}/{keyword_str}/{pca_dir}{c.n_comp}_cl/{thresh_dir}'
    

def create_file_prefix(c: T.clusterConfig, m: T.mappingConfig):
    """
    Creates file path name and prefix for plotting output
    """
 
    lon1, lon2 = 360 - m.lngRng[1], 360 - m.lngRng[0]
    dir_name = get_dir_path(m)
    date = pre.get_date(m.keywords, dir_name)
    lat_lon_str = f'{m.latRng[0]}-{m.latRng[1]}_{lon1}-{lon2}'
    keyword_str = '_'.join(m.keywords)
   
    return f'{date}_{keyword_str}_{lat_lon_str}_{c.n_comp}_sys_{m.cm_num}_'
