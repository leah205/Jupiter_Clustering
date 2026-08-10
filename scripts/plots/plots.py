
from config.config import cf
import numpy as np
import config.dicts as D
from scipy import linalg
import matplotlib as mpl
from scipy.stats import chi2


def create_cluster_plot(ax, keywords, index_x, index_y, input_arr, pred, n_comp,
                        cmap
                        ):
    """
    Purpose - Creates scatter plot of clusterings in two dimensions

    Parameters
    ----------
    ax, REQUIRED
        axis object of plot
    keywords: string[]
        - array of keywords for parameters in clustering
    index_x: int
        - index into keywords of independent variable
    index_y: int
        - index into keywords of dependent variable
    input_arr: np array
        - pixel radiances in parameters
    pred: np array
        - pixel cluster assignments
    n_comp: int
        - number of clusters
    
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
        ax.set_xlabel(D.keyword_dict[keywords[index_x]])
        ax.set_ylabel(D.keyword_dict[keywords[index_y]])
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

def plot_gmm_ellipsoids(ax, cluster_obj, cmap):
    """
    overlays ellipsoids capturing 95% total probability for each cluster over scatter plot

    Parameters
    -----------------------
    ax: matplotlib axis object
    cluster_obj: dict
        - return object of clustering algorthm including keys for cluster means and covariances
    cmap: color map of clusters

    
    """
    n_comp = cluster_obj["means"].shape[0]
  
    darker_colors = list(map(lambda c: (c[0] - 0.2, c[1] - 0.2, c[2] - 0.2), cmap.colors))
  
    for i, (mean, covar, color) in enumerate(zip(cluster_obj["means"], cluster_obj["covariances"], darker_colors[:n_comp])):
        v, w = linalg.eigh(covar)
        #captures 95% total probability
        k = np.sqrt(chi2.ppf(0.95, df=2))

        order = v.argsort()[::-1]
        v = v[order]
        w = w[:, order]
        v = k * np.sqrt(v)

        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        ell = mpl.patches.Ellipse(mean, 2 * v[0], 2 * v[1], angle=angle, edgecolor=color, facecolor = "none", linewidth = 2)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(1)
        ax.add_patch(ell)

