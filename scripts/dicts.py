from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# regions of interest







keyword_dict = {
    "NH3": "Ammonia Mole Fraction (ppm)",
    "PCld": "Cloud Pressure (mb)",
    "AOI": "Altitude Opacity Index",
    "CI": "Color Index",
    "Prob": "Posterior Probability"
}

color_dict = {
    "PCld": "Blues",
    "NH3": "terrain_r",
    "AOI": "viridis",
    "CI": "cividis",
    # set under black?
    "Prob": LinearSegmentedColormap.from_list("cluster_color", ["white", "yellow", "orange", "red"], N = 256)
}

ranges_dict = {
    "NH3": [0, 300],
    "PCld": [1000, 3100],
    "AOI": [0.1, 0.4],
    "CI": [0.3, 0.8],
    "Prob": [0, 1]
}

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


ROI_cmap = {
"Hot Spot": 'red',
                     "Gyre": "green",
                     "Cloud Plume": "blue",
                     "Reference": "black"
}
