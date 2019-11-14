import xarray as xr
import numpy as np
# import pyresample
from scipy.io import loadmat
from scipy.ndimage import binary_erosion
from numba import njit
from scipy.spatial import KDTree


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

        # TO DO: 4+ points scenarios
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


def load_kelly2013_data(matfile='slope_16.mat'):
    """
    load data from the mat file of Kelly et al., 2013
    doi:10.1002/grl.50872
    """

    rawdata = loadmat(matfile)
    nsections, nbounds = rawdata['slope']['lon'][0][0].shape
    _, nz = rawdata['slope']['z'][0][0].shape
    nmodes = rawdata['slope']['Nm'][0][0][0][0]
    dx = rawdata['slope']['dx'][0][0][0][0]
    z = rawdata['slope']['z'][0][0][0]
    # section has no natural value, set list of integers
    section = np.arange(nsections)
    mode = 1 + np.arange(nmodes)  # mode 0 is surface mode in litterature
    bounds = ['down', 'up']

    ds = xr.Dataset()
    ds['lon'] = xr.DataArray(data=rawdata['slope']['lon'][0][0],
                             coords={'cross_section': (['cross_section'],
                                                       section),
                                     'bounds': (['bounds'], bounds)},
                             dims=('cross_section', 'bounds'))
    ds['lon'].attrs = {'long_name': 'longitude', 'units': 'degrees_E'}

    ds['lat'] = xr.DataArray(data=rawdata['slope']['lat'][0][0],
                             coords={'cross_section': (['cross_section'],
                                                       section),
                                     'bounds': (['bounds'], bounds)},
                             dims=('cross_section', 'bounds'))
    ds['lat'].attrs = {'long_name': 'latitude', 'units': 'degrees_N'}

    ds['z'] = xr.DataArray(data=z,
                           coords={'z': (['z'], z)},
                           dims=('z'))
    ds['z'].attrs = {'long_name': 'depth', 'units': 'm downwards'}

    ds['H'] = xr.DataArray(data=rawdata['slope']['H'][0][0],
                           coords={'cross_section': (['cross_section'],
                                                     section),
                                   'bounds': (['bounds'], bounds)},
                           dims=('cross_section', 'bounds'))
    ds['H'].attrs = {'long_name': 'bathymetry', 'units': 'm upwards'}

    ds['N2'] = xr.DataArray(data=rawdata['slope']['N2'][0][0],
                            coords={'cross_section': (['cross_section'],
                                                      section),
                                    'z': (['z'], z)},
                            dims=('cross_section', 'z'))
    ds['N2'].attrs = {'long_name': 'Brunt-Vaisala frequency', 'units': 's-1'}

    ds['refl'] = xr.DataArray(data=rawdata['slope']['refl'][0][0],
                              coords={'cross_section': (['cross_section'],
                                                        section),
                                      'mode': (['mode'], mode)},
                              dims=('cross_section', 'mode'))
    ds['refl'].attrs = {'long_name': 'internal wave reflexion coef',
                        'units': 'nondim'}

    ds['trans'] = xr.DataArray(data=rawdata['slope']['trans'][0][0],
                               coords={'cross_section': (['cross_section'],
                                                         section),
                                       'mode': (['mode'], mode)},
                               dims=('cross_section', 'mode'))
    ds['trans'].attrs = {'long_name': 'internal wave transmission coef',
                         'units': 'nondim'}

    ds['dx'] = dx
    ds['dx'].attrs = {'long_name': 'horizontal resolution', 'units': 'm'}

    ds['Nm'] = mode
    ds['Nm'].attrs = {'long_name': 'vertical modes'}

    return ds


@njit
def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    # Compute spherical distance from spherical coordinates.
    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) +
           np.cos(phi1)*np.cos(phi2))
    arc = np.arccos(cos)
    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc


def find_closest_grid_cell_brute(lon, lat, longrid, latgrid):
    """
    find closest ocean cell of coordinates longrid/latgrid
    given a pair of lon/lat
    """

    distances = distance_on_unit_sphere(lat, lon, latgrid, longrid)
    ijmin = distances.argmin()
    jmin, imin = np.unravel_index(ijmin, longrid.shape)
    print(jmin, imin)
    return jmin, imin


def find_closest_grid_cell_kdtree(lon, lat, tree, nbpts=1):
    """ find the nbpts closest points from lon/lat in tree """
    if type(lon) == 'xr.core.dataarray.DataArray':
        lon = lon.values
    if type(lat) == 'xr.core.dataarray.DataArray':
        lat = lat.values
    query = tree.query([lon, lat], k=nbpts)
    return query


def build_kdtree(lon, lat):
    """ build the KDTree for collection of lon/lat points """
    if isinstance(lon, xr.core.dataarray.DataArray):
        lond = lon.values
    else:
        lond = lon
    if isinstance(lat, xr.core.dataarray.DataArray):
        latd = lat.values
    else:
        latd = lat
    tree = KDTree(list(zip(lond.ravel(), latd.ravel())))
    return tree


def query_to_model_indices(query, lonmodel):
    """ return distance and grid indices from query """
    distance, ravel_index = query
    j, i = np.unravel_index(ravel_index, lonmodel.shape)
    return distance, i, j


def change_lon_reference(lon_in, lon_ref):
    """ use periodicity to move lon_in in the range of lon_ref """
    for lon in lon_in:
        if lon > lon_ref.max():
            lon -= 360
        elif lon < lon_ref.min():
            lon += 360
        else:
            pass
    return lon_in
