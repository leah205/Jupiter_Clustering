# create a map for maximum posterior probability




# create a map for all pixels, probability of each cluster

def plot_posteriors(cluster, probs, ax):
    cluster_probs = probs[:, cluster]
    ax.scatter()