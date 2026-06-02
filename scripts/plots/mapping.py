from astropy.io import fits
import numpy as np
import pylab as pl
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as plots
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import scripts.dicts as D




def label_features(axis, LonLims, LatLims, ROI, showbands, is_cluster = False):
    #secaxis  = axis.secondary_yaxis('right')
    ylim = axis.get_ylim()
    if ROI:
        for R in ROI:
                axis.plot(np.array([ROI[R][2]+ROI[R][3],ROI[R][2]-ROI[R][3],
                              ROI[R][2]-ROI[R][3],ROI[R][2]+ROI[R][3],
                              ROI[R][2]+ROI[R][3]]),
                              90.-np.array([ROI[R][0],ROI[R][0],ROI[R][1],
                              ROI[R][1],ROI[R][0]]), color = D.ROI_cmap[R])
                
    # box = axis.get_position()
    
    belt = D.belt
    zone = D.zone
    # refactor
    bounds = [
        -39.6, -36.2, -32.4, -27.1, -19.7, -7.2, 6.9, 17.4, 24.2, 31.4, 35.4, 39.6
    ]

    ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1) ]

    lat_range_labels = ["SSTB", "STZ", "STB", "STrZ", "SEB", "EZ", "NEB", "NTrZ", "NTB",  "NTZ", "NNTB"]
    #secaxis.set_yticks(ticks)
    #secaxis.set_yticklabels(lat_range_labels)
  
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
    
    #secaxis.tick_params(axis='both', which='major', labelsize=9)
    axis.set_ylim(ylim)
    if(is_cluster):
        ROI_lines = []
        for R in ROI:
            ROI_lines.append( Line2D([0], [0], color = D.ROI_cmap[R], lw = 2))
        axis.legend(ROI_lines, list(ROI.keys()), fontsize = 8, loc = "upper left", bbox_to_anchor=(0.92, 1))
              
    return axis

def create_axis(axs3, LatLims, LonLims, LonSys, annotated = True):
    #fig3,axs3=pl.subplots(dpi=150, facecolor="white")
    axs3.grid(linewidth=0.2)
    axs3.ylim=[LatLims[0] ,LatLims[1]]
    axs3.xlim=[360-LonLims[0],360-LonLims[1]]
    #axs3.xlim=[LonLims[0],LonLims[1]]
    axs3.set_xticks(np.linspace(450,0,31), minor=False)
    yticks = np.linspace(LatLims[0], LatLims[1], 4)
   
    axs3.set_yticks(yticks, minor=False)
    axs3.set_yticklabels(yticks.astype(int))
    
    if(annotated):
        axs3.set_ylabel("Latitude (deg)",fontsize=10)
        axs3.set_xlabel("Sys. "+ str(LonSys) +" Longitude (deg)",fontsize=10)
        xticklabels=np.array(np.mod(np.linspace(450,0,31),360))
        axs3.set_xticklabels(xticklabels.astype(int))
        axs3.tick_params(axis='both', which='major', labelsize=9)
 
    axs3.set_adjustable('box') 
    return axs3

def plot_cluster_patch(config, patch, cmap, axis, n_comp, fig = None, cbarplot = True,  showbands = True):
   
   
    '''
    Purpose:
        to plot a patch of clusters with appropriate longitude/latitude scales
    '''  
    v_min, v_max = 0, n_comp
    n = v_max - v_min

    LatLims, LonLims = config.latRng, config.lngRng

    bounds = np.arange(v_min - 0.5, v_max + 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
   
    #np.nan_to_num(patch, copy=False, nan=-1.0, posinf=0.0, neginf=0.0)
    masked_patch = np.ma.masked_invalid(patch)
    cmap.set_bad("black")
    tx=np.linspace(v_max,v_min,n ,endpoint=True)
    
    show=axis.imshow(masked_patch,  origin='upper', interpolation = 'nearest', cmap = cmap, norm = norm,  
               extent=[360 - LonLims[0],360 - LonLims[1],90-LatLims[1],
                       90-LatLims[0]],
                       aspect="equal")
    print([90-LatLims[1],
                       90-LatLims[0]])
    create_axis(axis, [90 - LatLims[1], 90 - LatLims[0]], LonLims,  config.cm_num)

   
    axis.tick_params(axis='y', labelleft=True)
    axis = label_features(axis, LonLims, LatLims, config.ROI, showbands, True)
    
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]

 
    if cbarplot:
        cbar_title = "clusters"
        cbar = fig.colorbar(show, ticks = np.arange(v_max),  orientation='vertical',
                 ax=axis,fraction=0.046*im_ratio, pad=0.1)
        cbar.set_ticklabels(np.arange(1, v_max + 1))
        cbar.ax.set_ylabel(cbar_title, size=9, labelpad= 5, y = 0.5)#,labelpad=-20, y=0.5)


def plot_patch(config, patch, dim, axis, fig = None, cbarplot = True, cbar_title = "", annotated = True, showbands = True):
    LatLims, LonLims = config.latRng, config.lngRng
    cmap = D.color_dict[dim]
    title = D.keyword_dict[dim]
    cbar_reverse = False
    if(dim == "PCld"):
        cbar_reverse = True
    v_min, v_max = D.ranges_dict[dim][0], D.ranges_dict[dim][1]
    
    '''
    Purpose:
        to plot a patch with appropriate longitude/latitude scales
    '''  
    vn = v_min
    vx = v_max
    n = vx - vn
   # create_axis(axis, [LatLims[0] - 90, 90 - LatLims[1]], LonLims,  cm_num, annotated)
    np.nan_to_num(patch, copy=False, nan=-1.0, posinf=0.0, neginf=0.0)
    tx=np.linspace(vn,vx,5 ,endpoint=True)
    
    axis.set_title(title)
    show=axis.imshow(patch,  origin='upper', cmap = cmap, vmin=vn,vmax=vx,  
               extent=[360 - LonLims[0],360 - LonLims[1],90-LatLims[1],
                       90-LatLims[0]],
                       aspect="equal")
    
    create_axis(axis, [90 - LatLims[1], 90 - LatLims[0]], LonLims,  config.cm_num)
    
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]
    #axis.set_title(title, pad = 15)

    im_ratio = patch.shape[0]/patch.shape[1]
    if(annotated):
        axis = label_features(axis,  LonLims, LatLims, config.ROI, showbands)

    if cbarplot:
        
        cbar = fig.colorbar(show, ticks=tx, 
                   orientation='vertical',
                   ax=axis,fraction=0.046*im_ratio, pad=0.1)
        cbar.ax.set_yticklabels(np.around(tx,3))


        cbar.set_ticks(tx)
        cbar.set_ticklabels(np.around(tx, 3))
        cbar.ax.tick_params(labelsize=6,color="k")#if iSession >1:
        cbar.ax.set_ylabel(cbar_title, size=9,labelpad=5, y=0.5)

        if cbar_reverse:
            cbar.ax.invert_yaxis()
   





def reshape_clustered(indices, subset_shape, pred):
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







