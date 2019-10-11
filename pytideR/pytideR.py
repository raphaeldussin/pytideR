import xarray as xr
import numpy as np
# import pyresample
from scipy.ndimage import binary_erosion
from numba import njit


def find_costal_cells(mask):
    """ find the first coastal ocean cell by expanding land
    then substract original land mask.

    the mask passed should be the ocean mask (i.e. 1 on ocean/ 0 on land)
    we expand land (0) values by eroding ocean (1) values
    """
    # work on nx+1, ny+1 arrays to avoid boundary issues
    ny = len(mask['ny'])
    nx = len(mask['nx'])
    mask_w_halos = np.ones((ny+2, nx+2))
    mask_w_halos[1:-1, 1:-1] = mask.values
    expanded = binary_erosion(mask_w_halos)
    if expanded.sum() >= mask_w_halos.sum():
        raise ValueError("land expansion failed, check mask is ocean mask")

    coastdata = mask_w_halos - expanded
    coast = xr.zeros_like(mask)
    coast[:] = coastdata[1:-1, 1:-1]
    return coast


def define_angle(coastalcells, mask, periodic=True, northfold=True):
    """define the reflexion angle from the coastal cells computed by
    find_costal_cells and the ocean mask
    """

    jcoastalcells, icoastalcells = np.where(coastalcells >= 0.98)
    maskvalues = mask.values
    ny, nx = maskvalues.shape

    mask_extended = np.zeros((ny+2, nx+2))
    mask_extended[1:-1, 1:-1] = maskvalues[:, :]  # fill interior
    # E-W periodicity
    if periodic:
        mask_extended[1:-1, -1] = maskvalues[:, 0]
        mask_extended[1:-1, 0] = maskvalues[:, -1]
    else:
        mask_extended[1:-1, -1] = maskvalues[:, -1]
        mask_extended[1:-1, 0] = maskvalues[:, 0]
    # south pole (or boundary): repeat
    mask_extended[0, 1:-1] = maskvalues[0, :]
    # north pole (or boundary)
    if northfold:
        mask_extended[-1, 1:-1] = maskvalues[-1, ::-1]  # revert direction
    else:
        mask_extended[-1, 1:-1] = maskvalues[-1, :]  # repeat

    # we need to pass locations in the coord system of extended mask
    angledata = compute_angle(jcoastalcells+1, icoastalcells+1, mask_extended)

    angle = xr.zeros_like(mask)
    angle[:] = angledata[1:-1, 1:-1]
    return angle

# numba gets 4 times speedup on 1x1deg grid
@njit
def compute_angle(jcoastalcells, icoastalcells, mask_extended):
    """ compute core numba-friendly
    jcoastalcell, icoastalcell: numpy.array
    mask: numpy.array
    """

    nyp2, nxp2 = mask_extended.shape
    angledata = 1.e+20 * np.ones((nyp2, nxp2))

    # loop on all coastal cells
    for kcell in range(len(jcoastalcells)):
        icell = icoastalcells[kcell]
        jcell = jcoastalcells[kcell]

        # keep ben's notation, later change to explicit left, upperleft,...
        n1 = (mask_extended[jcell, icell+1] == 0)
        n2 = (mask_extended[jcell+1, icell+1] == 0)
        n3 = (mask_extended[jcell+1, icell] == 0)
        n4 = (mask_extended[jcell+1, icell-1] == 0)
        n5 = (mask_extended[jcell, icell-1] == 0)
        n6 = (mask_extended[jcell-1, icell-1] == 0)
        n7 = (mask_extended[jcell-1, icell] == 0)
        n8 = (mask_extended[jcell-1, icell+1] == 0)

        # strait coast scenarios
        if n1 and n2 and n8:
            angledata[jcell, icell] = 3 * np.pi / 2.
        elif n2 and n3 and n4:
            angledata[jcell, icell] = 0.
        elif n4 and n5 and n6:
            angledata[jcell, icell] = 1 * np.pi / 2.
        elif n6 and n7 and n8:
            angledata[jcell, icell] = 2 * np.pi / 2.
        # hard corners scenarios
        elif n1 and n3:
            angledata[jcell, icell] = 7 * np.pi / 4.
        elif n3 and n5:
            angledata[jcell, icell] = 1 * np.pi / 4.
        elif n5 and n7:
            angledata[jcell, icell] = 3 * np.pi / 4.
        elif n1 and n7:
            angledata[jcell, icell] = 5 * np.pi / 4.
        # mild corner scenarios (merid)
        elif n1 and n2:
            angledata[jcell, icell] = 13 * np.pi / 8.
        elif n4 and n5:
            angledata[jcell, icell] = 3 * np.pi / 8.
        elif n5 and n6:
            angledata[jcell, icell] = 5 * np.pi / 8.
        elif n1 and n8:
            angledata[jcell, icell] = 11 * np.pi / 8.
        # mild corner scenarios (zonal)
        elif n2 and n3:
            angledata[jcell, icell] = 15 * np.pi / 8.
        elif n3 and n4:
            angledata[jcell, icell] = 1 * np.pi / 8.
        elif n6 and n7:
            angledata[jcell, icell] = 7 * np.pi / 8.
        elif n7 and n8:
            angledata[jcell, icell] = 9 * np.pi / 8.
        # single point scenarios
        elif n1:
            angledata[jcell, icell] = 3 * np.pi / 2.
        elif n3:
            angledata[jcell, icell] = 0.
        elif n5:
            angledata[jcell, icell] = 1 * np.pi / 2.
        elif n7:
            angledata[jcell, icell] = 2 * np.pi / 2.
        else:
            raise ValueError('case not found')

    return angledata
