import matplotlib.pyplot as plt
import scripts.preprocessing.preprocessing as pre
from config.config import cf
from sklearn.metrics import silhouette_score
import scripts.clustering.clusters as CL
import numpy as np
from scipy.stats import gaussian_kde


keyword_dict = {
    "NH3": "Ammonia Mole Fraction (ppm)",
    "PCld": "Cloud Pressure (mb)",
    "AOI": "Altitude Opacity Index",
    "CI": "Color Index"
}

def create_cluster_plot(ax, keywords, index_x, index_y, input_arr, pred, n_comp,
                        cmap
                        ):
    """
    Purpose - Creates scatter plot of clusterings in two dimensions

    Parameters
    ----------
    ax, REQUIRED
        axis object of plot
    keywords, 
    
    -----------
    """
  
    cmap.set_under("black")
    if(keywords[index_y] == "PCld"):
        ax.yaxis.set_inverted(True)
    x = input_arr[:, index_x]
    y = input_arr[:, index_y]
    pred = np.array(pred)
    for cl in range(-1, n_comp):
        if(cl == -1 and cf["soft_clustering"] == False):
            continue
        mask = pred == cl
        x_cl = x[mask]
        y_cl = y[mask]
      
        
        cluster_label = f"{cl + 1}" if cl >= 0 else "NA"
        ax.scatter(x[mask], y[mask], s = 1, alpha = 0.02, color = cmap(cl), label = cluster_label)
      
        
        #ax.text(x_mean, y_mean, str(cl), fontsize = 10, zorder = 10)
        ax.set_xlabel(keyword_dict[keywords[index_x]])
        ax.set_ylabel(keyword_dict[keywords[index_y]])
        if(cl >= 0):
            x_mean = np.mean(x_cl)
            y_mean = np.mean(y_cl)
            x_std = np.std(x_cl)
            y_std = np.std(y_cl)
            ax.scatter(x_mean, y_mean, color = "black", s = 50, marker = 'x', zorder = 9)
            ax.annotate(
                str(cl + 1),
                (x_mean, y_mean),
                xytext=(8, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                fontsize=10,
                zorder = 10
           
                )
            print(f"{cluster_label}: mean=({x_mean:.2f}, {y_mean:.2f}), std=({x_std:.2f}, {y_std:.2f})")
 
        

    if(index_x == 0):
        l = ax.legend(markerscale = 10, loc = "upper left", bbox_to_anchor=(0.88, 1))
        for h in l.legendHandles:
            h.set_alpha(1)
            h.set_sizes([50])


    
    
    return 0

