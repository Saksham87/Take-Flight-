#Extracting data from OLYMPEX data

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fn = 'olympex_precip_reconstructed.nc'
ds = nc.Dataset(fn)
# print(ds['precipitation'].shape)
p=ds['precipitation'][1,:,:]

# print(p)
# print(type(p))

ds.variables.keys()

lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
time_var = ds.variables['time']
dtime = nc.num2date(time_var[:],time_var.units)
precip = ds.variables['precipitation'][:].flatten()
print(precip.shape)
rdtime=np.repeat(np.array([dtime]),5624,axis=1).flatten()
print(rdtime)
comb=np.dstack((precip,rdtime))
print(comb.shape)
mask = comb.all(axis=0)                   # Straight to 2D, no temp arrays
mask = np.logical_not(mask, out=mask)

nans = np.add.reduce(comb, axis=0)        # 2D temp array, not 3D
mask |= np.isnan(nans) 

# comb = comb[~np.isnan(comb)]
# comb = comb[comb!=0.0]

# rdtime=np.repeat(dtime,5624,axis=1).flatten()
# print(np.repeat(np.array([[1,2]]),3,axis=1).flatten())

# a pandas.Series designed for time series of a 2D lat,lon grid
precip_ts = pd.Series(mask) 

precip_ts.to_csv('precip.csv',index=False, header=True)



# prcp = ds['rainfall'][:]
# time = ds['time'][:]
# np.savetxt("data.csv",prcp,delimiter=",")
#print(prcp)
#plt.plot(time,prcp)