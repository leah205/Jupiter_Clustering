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
import matplotlib.patches as patches
from matplotlib.lines import Line2D


ROI_cmap = {
"Hot Spot": 'red',
                     "Gyre": "green",
                     "Cloud Plume": "blue",
                     "Reference": "black"
}

def label_features(axis, LonLims, LatLims, ROI, showbands, is_cluster = False):
    ylim = axis.get_ylim()
    if ROI:
        for R in ROI:
                axis.plot(np.array([ROI[R][2]+ROI[R][3],ROI[R][2]-ROI[R][3],
                              ROI[R][2]-ROI[R][3],ROI[R][2]+ROI[R][3],
                              ROI[R][2]+ROI[R][3]]),
                              90.-np.array([ROI[R][0],ROI[R][0],ROI[R][1],
                              ROI[R][1],ROI[R][0]]), color = ROI_cmap[R])
                
    # box = axis.get_position()
    
    belt={"SSTB":[-39.6,-36.2],
          "STB":[-32.4,-27.1],
          "SEB":[-19.7,-7.2],
          "NEB":[6.9,17.4],
          "NTB":[24.2,31.4],
          "NNTB":[35.4,39.6]}
    
    zone={"STZ":[-36.2,-32.4],
          "STrZ":[-27.1,-19.7],
          "EZ":[-7.2,6.9],
          "NTrZ":[17.4,24.2],
          "NTZ":[31.4,35.4]}

    bounds = [
        -39.6, -36.2, -32.4, -27.1, -19.7, -7.2, 6.9, 17.4, 24.2, 31.4, 35.4, 39.6
    ]

    ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1) ]

    lat_range_labels = ["SSTB", "STZ", "STB", "STrZ", "SEB", "EZ", "NEB", "NTrZ", "NTB",  "NTZ", "NNTB"]
    axis.set_yticks(ticks)
    axis.set_yticklabels(lat_range_labels)
  
    if showbands:
        for zb in belt:
            #print(zb,belt[zb])
            axis.fill_between([360-LonLims[0],360-LonLims[1]],[belt[zb][0],belt[zb][0]],[belt[zb][1],belt[zb][1]],
                                    color="0.5",alpha=0.25)
            axis.fill_between([360-LonLims[0],360-LonLims[1]],[belt[zb][0],belt[zb][0]],[belt[zb][1],belt[zb][1]],
                                    color="0.8",alpha=0.1)
        #axs1[1].annotate(zb,xy=[np.mean(belt[zb]),51],ha="center")
    #for zb in zone:
        #axs1[1].annotate(zb,xy=[np.mean(zone[zb]),51],ha="center")
    
    axis.tick_params(axis='both', which='major', labelsize=9)
    axis.set_ylim(ylim)
    if(is_cluster):
        ROI_lines = []
        for R in ROI:
            ROI_lines.append( Line2D([0], [0], color = ROI_cmap[R], lw = 2))


        axis.legend(ROI_lines, list(ROI.keys()), fontsize = 8, loc = "upper left", bbox_to_anchor=(0.92, 1))
              
    return axis

def create_axis(axs3, LatLims, LonLims, LonSys, annotated = True):
    #fig3,axs3=pl.subplots(dpi=150, facecolor="white")
    axs3.grid(linewidth=0.2)
    axs3.ylim=[LatLims[0] ,LatLims[1]]
    axs3.xlim=[360-LonLims[0],360-LonLims[1]]
    #axs3.xlim=[LonLims[0],LonLims[1]]
    axs3.set_xticks(np.linspace(450,0,31), minor=False)
    axs3.set_yticks(np.linspace(-45,45,7), minor=False)
    
    if(annotated):
        axs3.set_ylabel("Latitude (deg)",fontsize=10)
        axs3.set_xlabel("Sys. "+ str(LonSys) +" Longitude (deg)",fontsize=10)
        xticklabels=np.array(np.mod(np.linspace(450,0,31),360))
        axs3.set_xticklabels(xticklabels.astype(int))
        axs3.tick_params(axis='both', which='major', labelsize=9)
 
    axs3.set_adjustable('box') 
    return axs3

def plot_cluster_patch(patch, LatLims, LonLims, cmap, ROI, axis, v_min, v_max, title, cm_num, fig = None, cbarplot = True, cbar_title = "clusters", showbands = True):
    '''
    Purpose:
        to plot a patch of clusters with appropriate longitude/latitude scales
    '''  
    n = v_max - v_min

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
   
    axis = label_features(axis, LonLims, LatLims,  ROI, showbands, True)
    
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]

 
    if cbarplot:
 
        cbar = fig.colorbar(show, ticks = np.linspace(1, v_max, v_max),  orientation='vertical',
                 ax=axis,fraction=0.046*im_ratio, pad=0.2)
        cbar.ax.set_ylabel(cbar_title, size=12)#,labelpad=-20, y=0.5)


        cbar.ax.yaxis.set_label_coords(-1.5, 0.5)
        cbar.ax.yaxis.set_label_coords(-2.1, 0.5)
      


def plot_patch(patch, LatLims, LonLims, cmap, ROI, axis, v_min, v_max, title, cm_num, fig = None, cbarplot = True, annotated = True, cbar_title = "test", cbar_reverse = False, showbands = True):
    '''
    Purpose:
        to plot a patch with appropriate longitude/latitude scales
    '''  
    vn = v_min
    vx = v_max
    n = vx - vn
    create_axis(axis, [LatLims[0] - 90, 90 - LatLims[1]], LonLims,  cm_num, annotated)
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
    if(annotated):
        axis = label_features(axis,  LonLims, LatLims, ROI, showbands)

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
                           latRng, lngRng, cmap, ROI = {}, cm_num = 3, fig = None, cbar = False):
    indices = input_arr[:, input_arr.shape[1] - 1]
    cluster_map = create_cluster_arr(indices, subset_shape, pred)
    
    cmap.set_under("black")
    plot_cluster_patch(cluster_map, latRng, lngRng,  cmap, ROI, axis,  0, n_comp, "cluster map", cm_num, fig, cbar)





def create_cluster_arr(indices, subset_shape, pred):
    # takes index array and prediction array and reshapes into original shape with pixels at original indices
    
    subset_length = subset_shape[0] * subset_shape[1]
   
    #concatenat indexed and cluster array
    indexed_clusters = np.column_stack((indices, pred))
    oned_mapped_clusters = np.full(shape = subset_length, fill_value = np.nan)
    for r in range(indexed_clusters.shape[0]):
        index, cluster = int(indexed_clusters[r][0]), indexed_clusters[r][1]
        oned_mapped_clusters[index] = cluster
    mapped_clusters = oned_mapped_clusters.reshape(subset_shape)
    return mapped_clusters







