from astropy.io import fits
import numpy as np
import pylab as pl
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as plots
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from config.config import config
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_axis(axs3, LatLims, LonLims, LonSys):
    #fig3,axs3=pl.subplots(dpi=150, facecolor="white")
    axs3.grid(linewidth=0.2)
    axs3.ylim=[LatLims[0] ,LatLims[1]]
    axs3.xlim=[360-LonLims[0],360-LonLims[1]]
    #axs3.xlim=[LonLims[0],LonLims[1]]
    axs3.set_xticks(np.linspace(450,0,31), minor=False)
    xticklabels=np.array(np.mod(np.linspace(450,0,31),360))
    axs3.set_xticklabels(xticklabels.astype(int))
    axs3.set_yticks(np.linspace(-45,45,7), minor=False)
    axs3.tick_params(axis='both', which='major', labelsize=9)
    axs3.set_ylabel("Latitude (deg)",fontsize=10)
    axs3.set_xlabel("Sys. "+ str(LonSys) +" Longitude (deg)",fontsize=10)
 
    axs3.set_adjustable('box') 
    return axs3

def plot_cluster_patch(patch, LatLims, LonLims, cmap, axis, v_min, v_max, title, cm_num, fig = None, cbarplot = True, cbar_title = "test"):
    '''
    Purpose:
        to plot a patch of clusters with appropriate longitude/latitude scales
    '''  
    n = v_max - v_min
    print("patch being mapped:")
    print(patch)
    #bounds = [i for i in range(v_max - v_min + 1)]
    bounds = np.arange(v_min - 0.5, v_max + 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    create_axis(axis, [LatLims[0] - 90, 90 - LatLims[1]], LonLims,  cm_num)
    #np.nan_to_num(patch, copy=False, nan=-1.0, posinf=0.0, neginf=0.0)
    masked_patch = np.ma.masked_invalid(patch)
    cmap.set_bad("black")
    tx=np.linspace(v_max,v_min,n ,endpoint=True)
    

    show=axis.imshow(masked_patch,  origin='upper', interpolation = 'nearest', cmap = cmap, norm = norm,  
               extent=[360 - LonLims[0],360 - LonLims[1],90-LatLims[1],
                       90-LatLims[0]],
                       aspect="equal")
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]
    
    if cbarplot:
        cbar = fig.colorbar(show, ticks=tx, 
                   orientation='vertical',
                   ax=axis,fraction=0.046*im_ratio, pad=0.05)
        cbar.ax.set_yticklabels(np.around(tx,3))


        cbar.set_ticks(tx)
        cbar.set_ticklabels(np.around(tx, 3))
        cbar.ax.tick_params(labelsize=6,color="k")#if iSession >1:
        cbar.ax.set_ylabel(cbar_title, size=6)#,labelpad=-20, y=0.5)


        cbar.ax.yaxis.set_label_coords(-1.5, 0.5)
        cbar.ax.yaxis.set_label_coords(-2.1, 0.5)
        #cbar.remove()


def plot_patch(patch, LatLims, LonLims, cmap, axis, v_min, v_max, title, cm_num, fig = None, cbarplot = True, cbar_title = "test", cbar_reverse = False):
    '''
    Purpose:
        to plot a patch with appropriate longitude/latitude scales
    '''  
    vn = v_min
    vx = v_max
    n = vx - vn
    create_axis(axis, [LatLims[0] - 90, 90 - LatLims[1]], LonLims,  cm_num)
    np.nan_to_num(patch, copy=False, nan=-1.0, posinf=0.0, neginf=0.0)
    tx=np.linspace(vn,vx,5 ,endpoint=True)
    
    axis.set_title(title)
    show=axis.imshow(patch,  origin='upper', cmap = cmap, vmin=vn,vmax=vx,  
               extent=[360 - LonLims[0],360 - LonLims[1],90-LatLims[1],
                       90-LatLims[0]],
                       aspect="equal")
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]
    
    if cbarplot:
        
        cbar = fig.colorbar(show, ticks=tx, 
                   orientation='vertical',
                   ax=axis,fraction=0.046*im_ratio, pad=0.05)
        cbar.ax.set_yticklabels(np.around(tx,3))


        cbar.set_ticks(tx)
        cbar.set_ticklabels(np.around(tx, 3))
        cbar.ax.tick_params(labelsize=6,color="k")#if iSession >1:
        cbar.ax.set_ylabel(cbar_title, size=6)#,labelpad=-20, y=0.5)


        cbar.ax.yaxis.set_label_coords(-1.5, 0.5)
        cbar.ax.yaxis.set_label_coords(-2.1, 0.5)
        if cbar_reverse:
            cbar.ax.invert_yaxis()

def create_cluster_map(axis,  n_comp, pred, input_arr, subset_shape, 
                           latRng, lngRng, cmap, cm_num = 3, fig = None, cbar = False):
   
    cluster_map = create_cluster_arr(input_arr, subset_shape, pred)
    
    cmap.set_under("black")
    plot_cluster_patch(cluster_map, latRng, lngRng,  cmap, axis,  0, n_comp, "cluster map", cm_num, fig, cbar)





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
    return mapped_clusters







