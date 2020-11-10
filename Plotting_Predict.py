import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
import glob
from plotting_topo import plot_topo

def plot_topo(map,cmap=plt.cm.terrain,zorder=0,lonextent=(0,20),latextent=(35,60),plotstyle='pmesh'):
    '''
    -Utpal Kumar
    map: map object
    cmap: colormap to use
    lonextent: tuple of int or float
        min and max of longitude to use
    latextent: tuple of int or float
        min and max of latitude to use
    plotstyle: str
        use pmesh (pcolormesh) or contf (contourf)
    '''
    minlon,maxlon = lonextent
    minlat,maxlat = latextent
    minlon,maxlon = minlon-1,maxlon+1
    minlat,maxlat = minlat-1,maxlat+1
    #20 minute bathymetry/topography data
    etopo = np.loadtxt('topo/etopo20data.gz')
    lons  = np.loadtxt('topo/etopo20lons.gz')
    lats  = np.loadtxt('topo/etopo20lats.gz')
    # shift data so lons go from -180 to 180 instead of 20 to 380.
    etopo,lons = shiftgrid(180.,etopo,lons,start=False)
    lons_col_index = np.where((lons>minlon) & (lons<maxlon))[0]
    lats_col_index = np.where((lats>minlat) & (lats<maxlat))[0]
 
    etopo_sl = etopo[lats_col_index[0]:lats_col_index[-1]+1,lons_col_index[0]:lons_col_index[-1]+1]
    lons_sl = lons[lons_col_index[0]:lons_col_index[-1]+1]
    lats_sl = lats[lats_col_index[0]:lats_col_index[-1]+1]
    lons_sl, lats_sl = np.meshgrid(lons_sl,lats_sl)
    if plotstyle=='pmesh':
        cs = map.contourf(lons_sl, lats_sl, etopo_sl, 50,latlon=True,zorder=zorder, cmap=cmap,alpha=0.5, extend="both")
        limits = cs.get_clim()
        cs = map.pcolormesh(lons_sl,lats_sl,etopo_sl,cmap=cmap,latlon=True,shading='gouraud',zorder=zorder,alpha=1,antialiased=1,vmin=limits[0],vmax=limits[1],linewidth=0)
    elif plotstyle=='contf':
        cs = map.contourf(lons_sl, lats_sl, etopo_sl, 50,latlon=True,zorder=zorder, cmap=cmap,alpha=0.5, extend="both")
    return cs

# Sets the basemap boundaries - chosen for Atlantic Hurricane Region
lonmin, lonmax = 100, 60
latmin, latmax = 15, 45

# Setting up the figure for Mercator Project 
# Other projecting types can be found here: https://matplotlib.org/basemap/users/mapsetup.html
fig = plt.figure(figsize=(10,6))
axx = fig.add_subplot(111)
m = Basemap(projection='merc', resolution="f", llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax)

# Plots the topography on the basemap figure and draws the colorbar
cs = plot_topo(m,cmap=plt.cm.jet,zorder=2,lonextent=(lonmin, lonmax),latextent=(latmin, latmax))
fig.colorbar(cs, ax=axx, shrink=0.6)

# draw latitudinal and longitudinal grid lines with the step of 5 degrees
# We only show the labels on the left and bottom of the map.
m.drawcoastlines(color='k',linewidth=0.5,zorder=3)
m.drawcountries(color='k',linewidth=0.1,zorder=3)

parallelmin = int(latmin)
parallelmax = int(latmax)+1
m.drawparallels(np.arange(parallelmin, parallelmax,5,dtype='int16').tolist(),labels=[1,0,0,0],linewidth=0,fontsize=10, zorder=3)

meridianmin = int(lonmin)
meridianmax = int(lonmax)+1
m.drawmeridians(np.arange(meridianmin, meridianmax,5,dtype='int16').tolist(),labels=[0,0,0,1],linewidth=0,fontsize=10, zorder=3)