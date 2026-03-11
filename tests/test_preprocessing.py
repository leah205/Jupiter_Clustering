import scripts.preprocessing.preprocessing as pre
import numpy as np
def test_get_mapped_pix_arr():
    input_arr = [
        [1, 3],
        [0,0],
        [8,2]
    ]

    input_arr = np.array(input_arr)
    res = pre.get_mapped_pix_arr(input_arr)
    exp = [
        [1, 3, 0],
        [0,0, 1],
        [8,2, 2]
    ]
    exp = np.array(exp)

    np.testing.assert_array_equal(res, exp)

def test_get_filtered_pix_arr():
    input_arr = [
        [1, 3, 0],
        [0,0, 1],
        [8,2, 2]
    ]
    input_arr = np.array(input_arr)

    range_arr = [[0, 1], [3, 5]]
    res = pre.get_filtered_pix_arr(range_arr, input_arr)
    exp = np.array([
        [1, 3, 0]
    ])
    np.testing.assert_array_equal(res, exp)

