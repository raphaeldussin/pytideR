#!/usr/bin/env python
# coding: utf-8

# # Generating internal tides input files for MOM6 OM4p25 configuration

import xarray as xr
import pytideR
import matplotlib.pylab as plt
import numpy as np
import scipy.io
from sectionate import get_broken_line_from_contour

projectdir = '/local2/home/Sonya/Internal_tides_reflection'
#bendir = f'{projectdir}/from_ben/raytracing_deliverables/'
bendir = f"../data/"

griddir = '/archive/gold/datasets/OM5_025/coupled_mosaic_v20240410_unpacked/'
mask = xr.open_dataset(f'{griddir}/ocean_mask.nc')['mask']
bathy = xr.open_dataset(f'{griddir}/ocean_topog.nc')['depth']
geolon = xr.open_dataset(f'{griddir}/ocean_hgrid.nc')['x'][1::2,1::2]
geolat = xr.open_dataset(f'{griddir}/ocean_hgrid.nc')['y'][1::2,1::2]

ny, nx = geolon.shape
lonh = geolon.isel(nyp=int(ny/2.))
lath = geolat.isel(nxp=int(nx/4.))

#lonh = xr.open_dataset(f'{griddir}/ocean_geometry_OM_1deg.nc')['lonh']
#lath = xr.open_dataset(f'{griddir}/ocean_geometry_OM_1deg.nc')['lath']

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

#assert np.equal(jcoastalcell, jcoastalcell_ben).all()
#assert np.equal(icoastalcell, icoastalcell_ben).all()

angle = pytideR.define_angle(coast, mask)

angle_ben = xr.open_dataset(f'{bendir}/refl_angle_360x210_global.nc')['refl_angle']

# my algo resolves outer columns/rows but not ben's
#assert np.equal(angle.values[1:-1, 1:-1],
#                angle_ben.fillna(1.e+20).values[1:-1, 1:-1]).all()

check = False
if check:
    plt.figure(figsize=[12, 10])
    plt.pcolormesh(cnx, cny,
                   angle.where(angle != 1.e+20).values - angle_ben.values)
    plt.colorbar()
    plt.grid()
    plt.show()

# Actually that's great we're reproducing but we'd like the angles at 250 meters deep.
mask_250m = xr.ones_like(bathy)
mask_250m = mask_250m.where(bathy >= 250.).fillna(0.)

check = False
if check:
    plt.figure(figsize=[12, 10])
    plt.pcolormesh(mask_250m)
    plt.colorbar()
    plt.show()

isobath_250m = pytideR.find_costal_cells(mask_250m)
jcelliso250m, icelliso250m = np.where(isobath_250m >= 0.98)

angle_250m = pytideR.define_angle(isobath_250m, mask_250m)

check = False
if check:
    plt.figure(figsize=[12, 10])
    plt.pcolormesh(cnx, cny,
                   angle_250m.where(angle_250m != 1.e+20).values)
    plt.colorbar()
    plt.grid()
    plt.show()


#stop

# Other method: use contour directly

plt.figure()
C = plt.contour(bathy, [249, 251])
iseg, jseg = get_broken_line_from_contour(C, rounding='down', maxdist=3)
plt.close()

if check:
    plt.figure()
    plt.plot(iseg, jseg, 'ko')
    plt.title('bathy 250m points')
    #plt.show()

xiseg = xr.DataArray(iseg, dims=("pts"))
xjseg = xr.DataArray(jseg, dims=("pts"))
bathy_250m_verif = bathy[xjseg, xiseg]

#stop



# Loading slope coeficients from Sam Kelly
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/grl.50872
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

#plt.show()

#stop

ds_kelly = pytideR.compute_slope_midpoints(ds_kelly, geolon)

ds_kelly.to_netcdf('kelly_dataset.nc')

#stop

tree_model = pytideR.build_kdtree(geolon, geolat)

jmodel_dist, imodel_dist = pytideR.slope_midpoints_to_model_cells(ds_kelly,
                                                                  tree_model,
                                                                  geolon)

print(len(jmodel_dist), len(imodel_dist))
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

print(len(jmodel_cont), len(imodel_cont))

#stop

if check_model_pts:
    plt.figure()
    plt.pcolormesh(mask, cmap='binary_r')
    plt.scatter(imodel_cont, jmodel_cont,
                c=np.arange(len(imodel_cont)),
                cmap='nipy_spectral')
    plt.colorbar()
    plt.title('contiguous cells')

#jmodel_skimmed, imodel_skimmed = pytideR.skim_slope_cells(jmodel_cont,
#                                                          imodel_cont)

#if check_model_pts:
#    plt.figure()
#    plt.pcolormesh(mask, cmap='binary_r')
#    plt.grid()
#    plt.scatter(imodel_skimmed, jmodel_skimmed,
#                c=np.arange(len(imodel_skimmed)),
#                cmap='nipy_spectral')
#    plt.colorbar()
#    plt.title('contiguous+skimmed cells')
#    plt.show()


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


ds_out['refl_angle'] = pytideR.create_angle_array(imodel_cont, jmodel_cont,
                                                  bathy, spval=-999.9)

ds_out['refl_pref'] = ds_out['refl']

# test with ones
#ds_out['refl_pref'] = 1.0e+20 * xr.ones_like(ds_out['refl'])
#data = ds_out['refl_pref'].values
#data[np.where(ds_out['refl'].values < 2.)] = 1.
#ds_out['refl_pref'][:] = data

ds_out = ds_out.expand_dims(dim='TIME')
ds_out['mode'].encoding = {}
ds_out['TIME'] = xr.DataArray(data=np.array([0.]), dims=('TIME'))
ds_out['TIME'].attrs = {'units': "months since 0001-01-01 00:00:00",
                        'timaxe_origin': "01-JAN-0001 00:00:00",
                        'calendar': "NOLEAP",
                        'modulo': " ",
                        'axis': "T",
                        '_FillValue': 1.e+20}




ds_out['lonh'] = lonh
ds_out['lath'] = lath

ds_out = ds_out.rename({'lonh': 'LON', 'lath': 'LAT'})

ds_out['LON'].attrs.update({'units': "degrees_east",
                            'axis': "X",
                            '_FillValue': 1.e+20})
ds_out['LAT'].attrs.update({'units': "degrees_north",
                            'axis': "Y",
                            '_FillValue': 1.e+20})

ds_out['refl_pref'].attrs.update({'_FillValue': 1.e+20})
ds_out['refl_angle'].attrs.update({'_FillValue': -999.9})
ds_out['mode'].attrs.update({'_FillValue': 1.e+20})

ds_out = ds_out.isel(mode=0)
ds_out['refl_dbl'] = xr.zeros_like(ds_out['refl_pref'])

ds_out.to_netcdf('internal_waves_RTcoef_OM5p25.nc', unlimited_dims='TIME')
