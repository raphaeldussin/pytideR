import xarray as xr
import numpy as np
# import pyresample
from scipy.io import loadmat
from scipy.ndimage import binary_erosion
from numba import njit
from scipy.spatial import KDTree
from tqdm import tqdm


def find_costal_cells(mask):
    """ find the first coastal ocean cell by expanding land
    then substract original land mask.

    the mask passed should be the ocean mask (i.e. 1 on ocean/ 0 on land)
    we expand land (0) values by eroding ocean (1) values

    PARAMETERS:
    -----------

    mask: xarray.DataArray
        land sea mask

    RETURN:
    -------

    coast: np.array
        2d binary (1/0) mask of coastal cells
    """
    # work on nx+1, ny+1 arrays to avoid boundary issues
    ny, nx = mask.values.shape
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


def xr2npy(array):
    """return numpy.array from np or xr array"""
    if type(array) == xr.core.dataarray.DataArray:
        array = array.values
    return array


def find_closest_grid_cell_kdtree(lon, lat, tree, nbpts=1):
    """ find the nbpts closest points from lon/lat in tree """
    lon = xr2npy(lon)
    lat = xr2npy(lat)
    query = tree.query([lon, lat], k=nbpts)
    return query


def build_kdtree(lon, lat):
    """ build the KDTree for collection of lon/lat points """
    lon = xr2npy(lon)
    lat = xr2npy(lat)
    tree = KDTree(list(zip(lon.ravel(), lat.ravel())))
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


def compute_slope_midpoints(ds, lonmodel):
    """ compute slope midpoints from xarray.Dataset using
    lon, lat bounds """
    # compute mid-slope points
    ds['lon_mid'] = ds['lon'].mean(dim='bounds')
    ds['lat_mid'] = ds['lat'].mean(dim='bounds')
    # correct for periodicity based on model's longitude range
    ds['lon_mid'] = change_lon_reference(ds['lon_mid'], lonmodel)
    return ds


def slope_midpoints_to_model_cells(ds, tree_model, lonmodel,
                                   clonmid='lon_mid',
                                   clatmid='lat_mid',
                                   csection='cross_section'):
    """ for each slope in ds, find the closest model cell
    using a KDtree and return lists of [i,j] of found cells"""

    nsections = len(ds[clonmid])

    ilist = []
    jlist = []
    for k in range(nsections):
        q = find_closest_grid_cell_kdtree(ds[clonmid].isel({csection: k}),
                                          ds[clatmid].isel({csection: k}),
                                          tree_model, nbpts=1)
        d, i, j = query_to_model_indices(q, lonmodel)
        ilist.append(i)
        jlist.append(j)
    return jlist, ilist


def fill_between_cells(j1, i1, j2, i2, lonmodel, latmodel, cutoff=500000.):
    """ fill model cells between 2 distant model cells to get continuous
    segment """
    lonmodel = xr2npy(lonmodel)
    latmodel = xr2npy(latmodel)

    # get lon/lat of points
    if len(lonmodel.shape) == 2:
        lon1 = lonmodel[j1, i1]
        lon2 = lonmodel[j2, i2]
    elif len(lonmodel.shape) == 1:
        lon1 = lonmodel[i1]
        lon2 = lonmodel[i2]

    if len(latmodel.shape) == 2:
        lat1 = latmodel[j1, i1]
        lat2 = latmodel[j2, i2]
    elif len(latmodel.shape) == 1:
        lat1 = latmodel[i1]
        lat2 = latmodel[i2]

    # check against cutoff
    rearth = 6400000.  # meters
    dist = distance_on_unit_sphere(lat1, lon1, lat2, lon2) * rearth
    # filter off points close due to periodicity
    dlon = np.abs(lon2 - lon1)
    dlon_cutoff = cutoff * 360 / (2 * np.pi * rearth)

    jlist = []
    ilist = []
    # find intermediate points
    if (dist < cutoff) and (dlon < dlon_cutoff):
        # create a list of intermediate points and init to first point.
        jlist.append(j1)
        ilist.append(i1)
        iterate = True
        while iterate:
            # iterate until we arrive at the last point
            if (jlist[-1] == j2) and (ilist[-1] == i2):
                iterate = False
            else:
                # compute distance in i and j
                # and unit step (1 or -1) depending on
                # relative positions
                jpts = np.abs(j2 - jlist[-1])
                jstep = (j2 - jlist[-1]) / jpts
                ipts = np.abs(i2 - ilist[-1])
                istep = (i2 - ilist[-1]) / ipts
                # move one point in max(ipts, jpts)
                if jpts > ipts:
                    jlist.append(jlist[-1] + int(jstep))
                    ilist.append(ilist[-1])
                else:
                    jlist.append(jlist[-1])
                    ilist.append(ilist[-1] + int(istep))
    return jlist, ilist


def find_all_model_slope_cells(jlist_distantcells, ilist_distantcells,
                               lonmodel, latmodel, cutoff=500000.):
    """ fill in the gaps in the list of model cells on the slope """

    jlist_contigcells = []
    ilist_contigcells = []

    for k in tqdm(range(len(jlist_distantcells)-1)):
        j1, j2 = jlist_distantcells[k:k+2]
        i1, i2 = ilist_distantcells[k:k+2]
        jltmp, iltmp = fill_between_cells(j1, i1, j2, i2,
                                          lonmodel, latmodel,
                                          cutoff=cutoff)
        jlist_contigcells += jltmp
        ilist_contigcells += iltmp
    return jlist_contigcells, ilist_contigcells


def skim_slope_cells(jcells, icells):
    """ remove cells that are surrounded by 5 neighbors or more """
    jout = []
    iout = []
    npts = len(jcells)
    jicells = list(zip(jcells, icells))
    for k in range(npts):
        i = icells[k]
        j = jcells[k]
        ip1 = int(icells[k] + 1)
        im1 = int(icells[k] - 1)
        jp1 = int(jcells[k] + 1)
        jm1 = int(jcells[k] - 1)
        n_neighbors = 0
        # check 8 closest cells
        checked_neighbors = [(j, ip1), (jp1, ip1), (jp1, i), (jp1, im1),
                             (j, im1), (jm1, im1), (jm1, i), (jm1, ip1)]
        for neighbor in checked_neighbors:
            # print(neighbor)
            if neighbor in jicells:
                n_neighbors += 1
                # print(n_neighbors)
        if n_neighbors < 6:
            jout.append(j)
            iout.append(i)
    return jout, iout


def interpolate_midpoint_values_to_model_cells(data_midpoints,
                                               tree_midpoints,
                                               imodel, jmodel,
                                               lon_model, lat_model):
    """ interpolate data from cross-sections' midpoints to corresponding
    model cells found with the search and fill algorithms

    PARAMETERS:
    -----------

    data_midpoints: xarray.DataArray
        array of input data to interpolate on model grid
    tree_midpoints: KDTree
        tree of lon/lat positions of cross-sections
    imodel, jmodel: list of int
        model points that will receive a value
    lon_model, lat_model: xarray.DataArray or np.array
        2d lon/lat of the model grid

    RETURN:
    -------

    data_out: np.array
        the input data interpolated on model points
    """
    # inititialize output
    data_out = 1e+15 * np.ones(lon_model.shape)
    # zip j,i lists to have more compact indexing
    jimodel = list(zip(jmodel, imodel))

    nmodelpts = len(jimodel)
    for k in range(nmodelpts):
        # for each model point, find 2 closest slope points
        # d = distance, csections = index of cross-section
        d, csections = find_closest_grid_cell_kdtree(lon_model[jimodel[k]],
                                                     lat_model[jimodel[k]],
                                                     tree_midpoints, nbpts=2)
        # make weighted average
        wa = ((d[0] * data_midpoints.isel({'cross_section': csections[0]}) +
               d[1] * data_midpoints.isel({'cross_section': csections[1]})) /
              (d[0] + d[1]))
        data_out[jimodel[k]] = wa
    return data_out

def interpolate_midpoint_ds_to_model_cells(ds_midpoints,
                                           tree_midpoints,
                                           imodel, jmodel,
                                           lon_model, lat_model,
                                           variables=['refl', 'trans'],
                                           dims=('mode')):
    """ interpolate dataset of midpoints values onto model grid 
    
    PARAMETERS:
    -----------

    lon_model, lat_model: xarray.DataArray
    
    
    """

    #shape 
    out = xr.Dataset()
    out['lon'] = lon_model
    out['lat'] = lat_model


    return out
