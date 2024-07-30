import xarray as xr
import ipdb
#import utilities as ut
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib
import numpy as np
import cartopy.crs as ccrs

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')
import read_extract_data as red
import utilities as ut

if __name__ == "__main__":

    from importlib import reload

    qflux_name = ['tracmip_control']
    files = ['./tracmip_qflux.nc']
    fname ='tracmip_control'
    clevel =  np.linspace(-50,50,21)
    cmap = 'bwr'
    top_titles = ['Tracmip control']
    map_extents = {'left':-180, 'right':180, 'bottom':-90, 'top':90} 

    qflux_name=['Neale & Hoskins' , 'Stephen & Vallis', 'Hell et al']
    files = ['./perturbed_qflux_hoskins.nc', './perturbed_qflux_stephen.nc', './perturbed_qflux_hell.nc']
    fname = '75N_perturb_qflux'
    clevel = np.linspace(0.001,100,6) 
    cmap = 'Reds'
    map_extents = {'left':-100, 'right':100, 'bottom':20, 'top':90} 
    top_titles = ['Neale & Hoskins 2001' , 'Stephen & Vallis 2018', 'Hell et al. 2020']

    # Thesis figure
    qflux_name=['75N-A100', '60N-A52', '45N-A37', '15N-A27']
    files = ['./qflux_profiles/75NA100_stephen_qflux.nc', './qflux_profiles/60NA52_stephen_qflux.nc', './qflux_profiles/45NA37_stephen_qflux.nc', './qflux_profiles/15NA27_stephen_qflux.nc']
    qflux_name=['75N-A100', '45N-A100', '15N-A100']
    files = ['./qflux_profiles/75NA100_stephen_qflux.nc', './qflux_profiles/45NA100_stephen_qflux.nc', './qflux_profiles/15NA100_stephen_qflux.nc']
    fname = 'SV_perturb_qflux'
    clevel = np.linspace(0.001,100,6) 
    cmap_anom = ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
    cmap = matplotlib.colors.ListedColormap(cmap_anom)
    map_extents = {'left':-50, 'right':50, 'bottom':20, 'top':90} 
    top_titles = qflux_name

    # 2022-08-20
    qflux_name=['N75W45-A100', 'N45W16-A100', 'N15W12-A100']
    files = ['./qflux_profiles/75NA100_stephen_qflux.nc', './qflux_profiles/45NA100W16_stephen_qflux.nc',
            './qflux_profiles/15NA100W12_stephen_qflux.nc']

    # 2023-08-20
    qflux_name=['N75L-A100','N60L-A100', 'N45L-A100', 'N30L-A100', 'N15L-A100']
    files = ['./qflux_profiles/75NA100L_stephen_qflux.nc', './qflux_profiles/60NA100L_stephen_qflux.nc', 
            './qflux_profiles/45NA100L_stephen_qflux.nc','./qflux_profiles/30NA100L_stephen_qflux.nc',
            './qflux_profiles/15NA100L_stephen_qflux.nc']
    # 2024-March (for revision - high profiles fig)
    files = ['./highres_qflux_profiles/75NA100L_stephen_qflux.nc', './highres_qflux_profiles/60NA100L_stephen_qflux.nc', 
            './highres_qflux_profiles/45NA100L_stephen_qflux.nc','./highres_qflux_profiles/30NA100L_stephen_qflux.nc',
            './highres_qflux_profiles/15NA100L_stephen_qflux.nc']
    fname = 'SV_perturb_qflux'
    clevel = np.linspace(0.001,100,6) 
    clevel = [0,20,40,60,80,100]
    cmap_anom = ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
    cmap = matplotlib.colors.ListedColormap(cmap_anom)
    map_extents = {'left':-50, 'right':50, 'bottom':20, 'top':90} 
    top_titles = qflux_name

    ###############################################################

    ##################################################
    var = 'ocean_qflux'
    qflux = {}
    for i, q in enumerate(qflux_name):
        qflux_file = xr.open_dataset(files[i])
        qflux_temp = qflux_file.rename({'lon':'longitude', 'lat':'latitude'})[var].squeeze()
        extra_lon = qflux_temp.isel(longitude=-1) # append the last column with lons 360
        extra_lon = xr.DataArray(extra_lon.values.reshape(extra_lon.values.shape[0], -1), dims=['latitude', 'longitude'],
                            coords={'latitude':extra_lon.latitude, 'longitude':[360]})
        qflux[q] = xr.concat([qflux_temp,extra_lon], dim='longitude') 

    if False:
        # Plot the map
        rows = 1
        cols = len(qflux_name)
        map_grids = [qflux[q] for q in qflux_name]
        contour_map_grids = map_grids
        contour_map_grids = None
        shading_level_grids  = [clevel] * len(map_grids)
        contour_level_grids = shading_level_grids
        clabels_row = ['Wm-2'] * rows
        left_titles = ['']
        projection = ccrs.PlateCarree(); low_lat=-90
        xsize=3; ysize=2
        cmap_grids = [cmap] * len(map_grids)

    if True: # Plotting everything in the same map. (by summing them together). For the thesis figure
        rows = 1
        cols = 1
        #qflux['all'] = qflux['N75L-A100'] + qflux['N45L-A100'] + qflux['N15L-A100']
        #qflux['all'] = qflux['N60L-A100'] + qflux['N30L-A100'] 
        qflux['all'] = qflux['N75L-A100']
        qflux_name = ['all']
        map_extents = None
        map_extents = {'left':-60, 'right':60, 'bottom':0, 'top':90}
        map_grids = [qflux[q] for q in qflux_name]
        map_grids = [None for q in qflux_name]
        contour_map_grids = map_grids
        contour_map_grids = [qflux[q] for q in qflux_name]
        shading_level_grids  = [clevel] * len(map_grids)
        contour_level_grids = shading_level_grids
        projection = ccrs.PlateCarree()
        projection=ccrs.AlbersEqualArea(central_longitude=0.0, central_latitude=0)
        projection=ccrs.Mollweide(central_longitude=0)
        projection=ccrs.Orthographic(central_longitude=0.0, central_latitude=45)
        #projection=ccrs.NorthPolarStereo(true_scale_latitude=15,globe=True)
        projection=ccrs.EquidistantConic(false_northing=45)
        projection=ccrs.Stereographic(central_latitude=45, central_longitude=45, false_easting=0.0,
                false_northing=0.0, true_scale_latitude=45, globe=None)
        projection=ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90,globe=None)
        
        xsize=2.5; ysize=2.5; low_lat=0
        top_titles=['']
        cmap_grids = [cmap] * len(map_grids)
        clabels_row = ['Wm-2'] * rows
        left_titles = ['']
        contour_lw=1
    if False:
        lat_ticks = [-60,-30, 0, 30, 60]
        lon_ticks = [-120,-60,0,60,120]

        lat_ticks = [20,40,60,80]
        lon_ticks = [-30,0,30]

        lat_ticks = [37.5,45,52.5,60,67.5,75,82.5]
        lon_ticks = [-45,0,45]
    else:
        lat_ticks=None; lon_ticks=None
    # Do the plotting
    fig = plt.figure(figsize=(3,3))
    ax1 = fig.add_subplot(rows, cols, 1, projection=projection)
    ax1 = [ax1]
    ut.map_grid_plotting(map_grids,rows,cols,cmap_grids,shading_level_grids,clabels_row, top_titles=top_titles,
                left_titles=left_titles,projection=projection,xsize=xsize,ysize=ysize, low_lat=low_lat, gridline=True,
                coastlines=False, map_extents=map_extents, lat_ticks=lat_ticks, lon_ticks=lon_ticks, shading_extend='neither',
                contour_map_grids=contour_map_grids, contour_clevels=contour_level_grids, circle=False,
                pltf=fig, ax_all=ax1, colorbar=False,contour_lw=contour_lw)
    if True:
        clevel=contour_level_grids[0]
        contour_grid = qflux['N60L-A100']
        lats=contour_grid.latitude.values; lons=contour_grid.longitude.values
        ax1[0].contour(lons,lats,contour_grid,clevel,colors='red',linewidths=contour_lw,transform=ccrs.PlateCarree())
        contour_grid = qflux['N45L-A100']
        lats=contour_grid.latitude.values; lons=contour_grid.longitude.values
        ax1[0].contour(lons,lats,contour_grid,clevel,colors='blue',linewidths=contour_lw,transform=ccrs.PlateCarree())
        contour_grid = qflux['N30L-A100']
        lats=contour_grid.latitude.values; lons=contour_grid.longitude.values
        ax1[0].contour(lons,lats,contour_grid,clevel,colors='green',linewidths=contour_lw,transform=ccrs.PlateCarree())
        contour_grid = qflux['N15L-A100']
        lats=contour_grid.latitude.values; lons=contour_grid.longitude.values
        ax1[0].contour(lons,lats,contour_grid,clevel,colors='brown',linewidths=contour_lw,transform=ccrs.PlateCarree())




    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=500, pad_inches=0.01)

def calculate_the_ocean_transport():
	
	qflux_name = ['Meris_A50W16', 'Meris_A50W32', 'TracMIP']
	files = ['./merlis_schneider_50_16.nc', './merlis_schneider_50_32.nc', './tracmip_qflux.nc']

	heat_transport = {}
	latitudes = {}
	for i, q in enumerate(qflux_name):

		qflux_file = xr.open_dataset(files[i])
		qflux = qflux_file.rename({'lon':'longitude', 'lat':'latitude'})['ocean_qflux']
		qflux = qflux.isel(longitude=0) # this is zonal symmetric
		qflux = qflux * -1 # inverted

		if False: ### Do interpolation here
			interpolate_latitude = np.arange(-90,90,0.5)
			# find the location of 0
			idx0= np.where(interpolate_latitude==0)[0][0]
			interpolate_latitude[idx0-1] = -0.00000001
			interpolate_latitude[idx0+1] = 0.00000001
			idx0= np.where(interpolate_latitude==0)[0][0]
			interpolate_latitude = np.delete(interpolate_latitude, idx0) # Remove the 0 latitude
			qflux = qflux.interp(latitude=interpolate_latitude)
			qflux = qflux.dropna(dim='latitude')

		# set the area
		latitudes[q] = qflux.latitude.values  # Keep the 'real' latitude
		lat_band = np.diff(latitudes[q]).mean()
		area_bands = lat_band * 111200 * 2*np.pi*6371000 * np.cos(latitudes[q]*np.pi/180)
		# area should be the cumulative sum
		areas = np.cumsum(area_bands)
		# put the area back into the q-qflux as the coordinate
		qflux = qflux.assign_coords(latitude=areas)

		heat_transport[q] = []
		for i, area in enumerate(areas):

			qflux_temp = qflux.sel(latitude=slice(0,area))
			qflux_temp_integrate = qflux_temp.integrate('latitude')
			heat_transport[q].append(qflux_temp_integrate.values/1E15)
	

	# Plot the timeseries
	timeseries = [heat_transport[q] for q in qflux_name]
	times = [latitudes[q] for q in qflux_name]
	ylabel = 'Energy transport (PW)'
	xlabel = 'latitude'
	# Plot the x versus y
	fname = 'Meridional_ocean_transport'
	legend_label = qflux_name
	line_colors = ['k', 'r' ,'g'][0:len(timeseries)]
	xlims = (-90,90)
	ylims = (-3.5,3.5)
	ut.timeseries_plotting(times, timeseries, ylabel, fname=fname, xlabel=xlabel,
				zero_hline=True, legend_label=legend_label,colors=line_colors,grid=True
				,xlims=xlims,ylims=ylims,xsize=6,ysize=3)


#calculate_the_ocean_transport()
#plotting_perturb_qflux()
