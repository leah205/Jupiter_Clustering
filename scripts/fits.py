import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU

import scripts.plots.mapping as MP


def create_cluster_fits(
    reshaped_pred: npt.NDArray[np.float64],
    probs: npt.NDArray[np.float64],
    n_comp: int,
    subset_shape,
    indices,
) -> HDUList:
    clus_assignments_hdu = fits.PrimaryHDU(data=reshaped_pred)
    print("reshaped pred:", reshaped_pred.shape)
    hdulist: list[PrimaryHDU | fits.ImageHDU] = [clus_assignments_hdu]
    for i in range(n_comp):
        cluster_probs = probs[:, i]
        reshaped_probs = MP.reshape_clustered(cluster_probs, indices, subset_shape)
        cluster_probs_hdu = fits.ImageHDU(data=reshaped_probs)
        hdulist.append(cluster_probs_hdu)

    hdul = fits.HDUList(hdulist)
    return hdul
