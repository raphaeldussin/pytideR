#!/usr/bin/env python
# coding: utf-8

# # Generating internal tides input files for MOM6 1 degree grid configuration

import xarray as xr
import pytideR
import matplotlib.pylab as plt
import numpy as np
import scipy.io

projectdir = '/home/Raphael.Dussin/Sonya/Internal_tides_reflection/'
bendir = f'{projectdir}/from_ben/raytracing_deliverables/'

griddir = '/home/Raphael.Dussin/Sonya/Internal_tides_reflection/pytideR/data/'
mask = xr.open_dataset(f'{griddir}/ocean_mask_OM_1deg.nc')['mask']
geolon = xr.open_dataset(f'{griddir}/ocean_geometry_OM_1deg.nc')['geolon']
geolat = xr.open_dataset(f'{griddir}/ocean_geometry_OM_1deg.nc')['geolat']

coast = pytideR.find_costal_cells(mask)
jcoastalcell, icoastalcell = np.where(coast >= 0.98)

# NB convention: i,j are cell centers
# pcolormesh needs N+1 coordinates points to plot properly between indices
# we want full indices to be cell centers so land points will go from side
# to side, land points needs offset
ny, nx = mask.values.shape
cnx = np.arange(nx+1) - 0.5
cny = np.arange(ny+1) - 0.5

# Plot for visual check
check = False
if check:
    plt.figure(figsize=[12, 10])
    plt.pcolormesh(cnx, cny, mask)
    plt.grid()
    plt.plot(icoastalcell, jcoastalcell, 'ro')
    plt.show()

# Compare with Ben's results from matlab algo
# and we get identical results

filecoastben = f'{bendir}/coast_data.mat'
coastben = scipy.io.loadmat(filecoastben)['coast']
jcoastalcell_ben, icoastalcell_ben = np.where(coastben >= 0.98)

assert np.equal(jcoastalcell, jcoastalcell_ben).all()
assert np.equal(icoastalcell, icoastalcell_ben).all()

angle = pytideR.define_angle(coast, mask)

angle_ben = xr.open_dataset(f'{bendir}/refl_angle_360x210_global.nc')['refl_angle']

# my algo resolves outer columns/rows but not ben's
assert np.equal(angle.values[1:-1, 1:-1],
                angle_ben.fillna(1.e+20).values[1:-1, 1:-1]).all()

check = False
if check:
    plt.figure(figsize=[12, 10])
    plt.pcolormesh(cnx, cny,
                   angle.where(angle != 1.e+20).values - angle_ben.values)
    plt.colorbar()
    plt.grid()
    plt.show()

# Loading slope coeficients from Sam Kelly
ds_kelly = pytideR.load_kelly2013_data(matfile=f'{projectdir}/pytideR/data/slope_16.mat')

# exploring the data
explore = False
if explore:
    plt.figure()
    plt.scatter(ds_kelly['lon'].sel(bounds='down'),
                ds_kelly['lat'].sel(bounds='down'),
                color='tab:blue', marker='o')
    plt.scatter(ds_kelly['lon'].sel(bounds='up'),
                ds_kelly['lat'].sel(bounds='up'),
                color='tab:orange', marker='o')

    plt.figure()
    plt.scatter(ds_kelly['lon'].sel(bounds='down'),
                ds_kelly['lat'].sel(bounds='down'),
                c=ds_kelly['H'].sel(bounds='down'), cmap='terrain_r')
    plt.scatter(ds_kelly['lon'].sel(bounds='up'),
                ds_kelly['lat'].sel(bounds='up'),
                c=ds_kelly['H'].sel(bounds='up'), cmap='terrain_r')
    plt.colorbar()

    plt.figure()
    plt.scatter(ds_kelly['lon'].sel(bounds='down'),
                ds_kelly['lat'].sel(bounds='down'),
                c=np.arange(len(ds_kelly['lon'].sel(bounds='down'))),
                cmap='nipy_spectral')
    plt.scatter(ds_kelly['lon'].sel(bounds='up'),
                ds_kelly['lat'].sel(bounds='up'),
                c=np.arange(len(ds_kelly['lon'].sel(bounds='down'))),
                cmap='nipy_spectral')
    plt.colorbar()


ds_kelly = pytideR.compute_slope_midpoints(ds_kelly, geolon)

tree_model = pytideR.build_kdtree(geolon, geolat)

jmodel_dist, imodel_dist = pytideR.slope_midpoints_to_model_cells(ds_kelly,
                                                                  tree_model,
                                                                  geolon)

check_model_pts = True

if check_model_pts:
    plt.figure()
    plt.pcolormesh(mask, cmap='binary_r')
    plt.scatter(imodel_dist, jmodel_dist,
                c=np.arange(len(imodel_dist)),
                cmap='nipy_spectral')
    plt.colorbar()
    plt.title('non-contiguous cells')

jmodel_cont, imodel_cont = pytideR.find_all_model_slope_cells(jmodel_dist,
                                                              imodel_dist,
                                                              geolon,
                                                              geolat,
                                                              cutoff=500000.)

if check_model_pts:
    plt.figure()
    plt.pcolormesh(mask, cmap='binary_r')
    plt.scatter(imodel_cont, jmodel_cont,
                c=np.arange(len(imodel_cont)),
                cmap='nipy_spectral')
    plt.colorbar()
    plt.title('contiguous cells')

jmodel_skimmed, imodel_skimmed = pytideR.skim_slope_cells(jmodel_cont,
                                                          imodel_cont)

if check_model_pts:
    plt.figure()
    plt.pcolormesh(mask, cmap='binary_r')
    plt.grid()
    plt.scatter(imodel_skimmed, jmodel_skimmed,
                c=np.arange(len(imodel_skimmed)),
                cmap='nipy_spectral')
    plt.colorbar()
    plt.title('contiguous+skimmed cells')
    plt.show()


# --- mapping of slope data onto model cells:

tree_midpoints = pytideR.build_kdtree(ds_kelly['lon_mid'], ds_kelly['lat_mid'])

# RD: this should be done on skimmed cells
mapping = pytideR.create_mapping_midpoints_to_model_cells(tree_midpoints,
                                                          imodel_cont,
                                                          jmodel_cont,
                                                          geolon,
                                                          geolat)

ds_out = pytideR.interpolate_slope_dataset_to_model_cells(ds_kelly, mapping,
                                                          geolon, geolat)


print(ds_out)

ds_out.to_netcdf('internal_waves_RTcoef_1deg.nc')
