from astropy.io import fits
import numpy as np
import pylab as pl
import scripts.preprocessing.preprocessing as pre
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from config.config import config

def createLonMap(LonLims, shape):
    #oned_map = np.linspace(LonLims[0], LonLims[1], shape[1])
    #res = np.tile(oned_map, (shape[0], 1))
    print("hi")


def createLatMap(LatLims, shape):
    #oned_map =  np.lonspace(LatLims[0], LatLims[1], shape[0])
    #res = np.tile(oned_map, (1, shape[1]))
    print("yo")

def create_axis(LonLims, LatLims, Ltitle, LonSys):
    #fig3,axs3=pl.subplots(1,2,figsize=(figxy[0],figxy[1]), dpi=150, facecolor="white")
    #fig3.suptitle(suptitle,x=0.5,ha='center',color='k')
    #fig3.suptitle(mapyhdr["DATE-OBS"].replace("_"," ")+", CM"+LonSys+"="
    #              +str(int(PlotCM)),x=0.5,ha='center',color='k')
    fig3, axs3 = pl.subplots()

    axs3[0].grid(linewidth=0.2)
    axs3[0].ylim=[-45.,45.]
    axs3[0].xlim=[360-LonLims[0],360-LonLims[1]]
    axs3[0].set_xticks(np.linspace(450,0,31), minor=False)
    xticklabels=np.array(np.mod(np.linspace(450,0,31),360))
    axs3[0].set_xticklabels(xticklabels.astype(int))
    axs3[0].set_yticks(np.linspace(-45,45,7), minor=False)
    axs3[0].tick_params(axis='both', which='major', labelsize=9)
    axs3[0].set_ylabel("Planetographic Latitude (deg)",fontsize=10)
    axs3[0].set_xlabel("Sys. "+LonSys+" Longitude (deg)",fontsize=10)
    axs3[0].set_title(Ltitle,fontsize=10)
    axs3[0].set_adjustable('box') 
    axs3[1].set_adjustable('box')
    return(axs3)

def plot_patch(patch, LatLims, LonLims, axis, colorscale, vn = 0.1, vx = 0.2, n = 6):
    '''
    Purpose:
        to plot a patch with appropriate longitude/latitude scales
    '''
    np.nan_to_num(patch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    tx=np.linspace(vn,vx,n,endpoint=True)
    show=axis.imshow(patch, colorscale, origin='upper',vmin=vn,vmax=vx,  
               extent=[360-LonLims[0],360-LonLims[1],90-LatLims[1],
                       90-LatLims[0]],
                       aspect="equal")

    im_ratio = patch.shape[0]/patch.shape[1]

    
def create_cluster_map(keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 360], cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    print("silhouette score: " + str(silhouette_score(input_arr, pred)))
    cluster_map = create_cluster_arr(input_arr, subset_shape, pred)
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul[0].data = cluster_map
    #mp.createLonMap(lngRng, hdul[0].shape[1])
    hdul.writeto(f'{config["output"]}/cluster_maps/spatial_map_{cov_type}_{n_comp}.fits', overwrite= True)

def create_cluster_arr(input_arr, subset_shape, pred):
    subset_length = subset_shape[0] * subset_shape[1]
    indices = input_arr[:, input_arr.shape[1] - 1]
    #concatenat indexed and cluster array
    indexed_clusters = np.column_stack((indices, pred))
    oned_mapped_clusters = np.full(shape = subset_length, fill_value = np.nan)
    for r in range(indexed_clusters.shape[0]):
        index, cluster = int(indexed_clusters[r][0]), int(indexed_clusters[r][1])
        oned_mapped_clusters[index] = cluster
    mapped_clusters = oned_mapped_clusters.reshape(subset_shape)

create_cluster_map(["NH3", "PCld"], [[30, 250], [1000, 2500]])