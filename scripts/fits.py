import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU


def create_cluster_fits(
    reshaped_pred: npt.NDArray[np.float64],
    prob_map: npt.NDArray[np.float64],
    n_comp: int,
) -> HDUList:
    clus_assignments_hdu = fits.PrimaryHDU(data=reshaped_pred)
    print("reshaped pred:", reshaped_pred.shape)
    hdulist: list[PrimaryHDU | fits.ImageHDU] = [clus_assignments_hdu]
    for i in range(n_comp):
        cluster_probs_hdu = fits.ImageHDU(data=prob_map[i, :])
        hdulist.append(cluster_probs_hdu)

    hdul = fits.HDUList(hdulist)
    return hdul
