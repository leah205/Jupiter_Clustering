import matplotlib.pyplot as plt
from scripts.preprocessing.preprocessing import get_parameter_nd_array
import numpy as np


def create_plot(input_array, param1, param2):
    print(input_array.ndim)
    print(input_array.shape)
    if not isinstance(input_array, np.ndarray) or input_array.ndim != 3:
        raise TypeError("input should be 3d array")
    x = input_array[0].flatten()
    y = input_array[1].flatten()
   # flattened_arr = input_array.reshape(2, -1)
    mask =  ~np.isnan(x) & ~np.isnan(y)
 
    plt.scatter(x[mask], y[mask], s = 1)
    plt.xlabel(param1)
    plt.ylabel(param2)
    s = 1
    plt.show()

create_plot(get_parameter_nd_array(["NH3", "PCld"]), "ammonia content", "cloud pressure")

