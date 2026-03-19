from astropy.io import fits
import numpy as np
import pylab as pl
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from config.config import config
import scripts.plots.mapping as MP
from matplotlib.colors import ListedColormap


def output_cluster_map_and_plot(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 50], cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)

    fig, axis = pl.subplots(2, 1)
    MP.create_cluster_map(axis[0], date, n_comp, pred, input_arr, subset_shape, sil_score, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1], date, keywords, input_arr, pred, sil_score, n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/{date}_{lat_lon_str}_{n_comp}.png')

def output_cluster_map(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 50], cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)
    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, date, n_comp, pred, input_arr, subset_shape, sil_score, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/cluster_maps/spatial_map_{date}_{lat_lon_str}_{n_comp}.png')


def output_cluster_plot(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 50], cm_num = 3):
    
   
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    print(pred)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)
    fig, axis = pl.subplots(1, 1)
    PL.create_cluster_plot(axis, date, keywords, input_arr, pred, sil_score, n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/cluster_plots/clusters_{date}_{lat_lon_str}_{cov_type}_{n_comp}_.png')


def plot_patch(patch,LatLims,LonLims, colorscale,axis,
               cbarplot=True,cbar_title="Test",cbar_reverse=False,vn=0.10,vx=0.20,n=6):
    """
    Purpose:
        To plot a map patch with appropriate latitude and longitude scales,
        and, optionally, a color bar

    Parameters
    ----------
    patch : TYPE
        DESCRIPTION.
    LatLims : TYPE
        DESCRIPTION.
    LonLims : TYPE
        DESCRIPTION.
    CM2 : TYPE
        DESCRIPTION.
    LonRng : TYPE
        DESCRIPTION.
    colorscale : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.
    cbarplot : TYPE, optional
        DESCRIPTION. The default is True.
    cbar_title : TYPE, optional
        DESCRIPTION. The default is "Test".
    cbar_reverse : TYPE, optional
        DESCRIPTION. The default is False.
    vn : TYPE, optional
        DESCRIPTION. The default is 0.10.
    vx : TYPE, optional
        DESCRIPTION. The default is 0.20.
    n : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    None.

    """
    import numpy as np
    import pylab as pl

    np.nan_to_num(patch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    tx=np.linspace(vn,vx,n,endpoint=True)
    #vmin=vn,vmax=vx,
    show=axis.imshow(patch, colorscale, origin='upper',  
               extent=[360-LonLims[0],360-LonLims[1],90-LatLims[1],
                       90-LatLims[0]],
                       aspect="equal")

    im_ratio = patch.shape[0]/patch.shape[1]
    if cbarplot:
        cbar = pl.colorbar(show, ticks=tx, 
                   orientation='vertical',cmap='gist_heat',
                   ax=axis,fraction=0.046*im_ratio, pad=0.05)
        cbar.ax.set_yticklabels(np.around(tx,3))
        cbar.ax.tick_params(labelsize=6,color="k")#if iSession >1:
        cbar.ax.set_ylabel(cbar_title,size=6)#,labelpad=-20, y=0.5)
        #cbar.ax.yaxis.set_label_coords(-1.5, 0.5)
        cbar.ax.yaxis.set_label_coords(-2.1, 0.5)
        if cbar_reverse:
            cbar.ax.invert_yaxis()

    return patch,vn,vx,tx



def create_map_comparison(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 50], cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)
    fig, axis = pl.subplots(3, 1)
    cloud_map = pre.get_patch("PCld", latRng, lngRng)
    amm_map = pre.get_patch("NH3", latRng, lngRng)
    MP.create_cluster_map(axis[0], date, n_comp, pred, input_arr, subset_shape, sil_score, latRng, lngRng, cmap, cm_num)
    plot_patch(cloud_map, latRng, lngRng, "Wistia", axis[1])
    plot_patch(amm_map, latRng, lngRng, "autumn", axis[2])
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/{date}_{lat_lon_str}_{n_comp}_comparison.png')
    


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
create_map_comparison("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 4)#


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#output_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
    