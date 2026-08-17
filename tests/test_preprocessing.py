import scripts.preprocessing.preprocessing as pre
import scripts.plots.mapping as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")


def test_plot_longitude_extent_uses_configured_range():
    lon_lims = [75, 105]
    assert mp.get_plot_lon_extent(lon_lims) == [75, 105]


def test_roi_longitude_is_normalized_to_map_frame():
    assert mp.normalize_roi_lon_to_map_frame(267, [75, 105]) == 93.0
    assert mp.normalize_roi_lon_to_map_frame(14, [330, 360]) == 346.0


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

