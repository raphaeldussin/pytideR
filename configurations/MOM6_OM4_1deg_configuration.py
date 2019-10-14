#!/usr/bin/env python
# coding: utf-8

# # Generating internal tides input files for MOM6 1 degree grid configuration

import xarray as xr
import pytideR
import matplotlib.pylab as plt
import numpy as np
import scipy.io 

mask = xr.open_dataset('../data/ocean_mask_OM_1deg.nc')['mask']

coast = pytideR.find_costal_cells(mask)
jcoastalcell, icoastalcell = np.where(coast >= 0.98)

# NB convention: i,j are cell centers
# pcolormesh needs N+1 coordinates points to plot properly between indices
# we want full indices to be cell centers so land points will go from side to side 
# land points needs offset
nx = len(mask['nx'])
ny = len(mask['ny'])
cnx = np.arange(nx+1) - 0.5
cny = np.arange(ny+1) - 0.5

# Plot for visual check

#plt.figure(figsize=[12,10])
#plt.pcolormesh(cnx, cny, mask)
#plt.grid()
#plt.plot(icoastalcell, jcoastalcell, 'ro')
#plt.show()

# Compare with Ben's results from matlab algo
# and we get identical results

filecoastben = '../../from_ben/raytracing_deliverables/coast_data.mat'
coastben = scipy.io.loadmat(filecoastben)['coast']
jcoastalcell_ben, icoastalcell_ben = np.where(coastben >= 0.98)

assert np.equal(jcoastalcell, jcoastalcell_ben).all()
assert np.equal(icoastalcell, icoastalcell_ben).all()

angle = pytideR.define_angle(coast, mask)

angle_ben = xr.open_dataset('/home/Raphael.Dussin/Sonya/Internal_tides_reflection/from_ben/raytracing_deliverables/refl_angle_360x210_global.nc')['refl_angle']

# my algo resolves outer columns/rows but not ben's
assert np.equal(angle.values[1:-1,1:-1], angle_ben.fillna(1.e+20).values[1:-1,1:-1]).all()

plt.figure(figsize=[12,10])
plt.pcolormesh(cnx, cny, angle.where(angle != 1.e+20).values - angle_ben.values)
plt.colorbar()
plt.grid()
#plt.plot(icoastalcell, jcoastalcell, 'ro')
plt.show()

