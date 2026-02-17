import matplotlib.pyplot as plt
from scripts.preprocessing.preprocessing import get_parameter_nd_array
import numpy as np
from scipy.stats import gaussian_kde

#are the axes ranges even right?
#get denser regions
def create_plot(input_array, param1, param2):
    print(input_array.ndim)
    print(input_array.shape)
    if not isinstance(input_array, np.ndarray) or input_array.ndim != 3:
        raise TypeError("input should be 3d array")
    x = input_array[0].flatten()
    


    y = input_array[1].flatten()
   # flattened_arr = input_array.reshape(2, -1)
    mask =  ~np.isnan(x) & ~np.isnan(y)
    x_vals, y_vals = x[mask], y[mask]
    xy = np.vstack([x_vals,y_vals])
    z = gaussian_kde(xy)(xy)
    ax = plt.gca()
    ax.set_xlim([100, 400])
    ax.set_ylim([1500, 2200])
    x1 = [0, 1, 2]
    y1 = [1, 3, 5]
    plt.scatter(x_vals, y_vals, c = z, s = 1)
    plt.xlabel(param1)
    plt.ylabel(param2)
    s = 1
    plt.show()


create_plot(get_parameter_nd_array(["NH3", "PCld"]), "ammonia content", "cloud pressure")

