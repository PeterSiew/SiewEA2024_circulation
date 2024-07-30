import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import xarray as xr
import cartopy.crs as ccrs
from importlib import reload
import ipdb

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')
import read_extract_data as red
import utilities as ut


if __name__ == "__main__":

    if False: # cold things - ready to deal
        exp_control = 'gorman_tracmip_control'; exp_perturb = '45NA100W16_qflux'
        exp_control = 'gorman_tracmip_control'; exp_perturb = '15NA100W12_qflux'
        exp_control = 'gorman_tracmip_control'; exp_perturb = '75NA100_qflux'
        exp_perturb = '75NA100_qflux_rerun' # Re-run has all levels
        var = 'omegatheta_anom'
        vars = ['sphum']
        vars = ['ucompvcomp_anom']; vars = ['ucompucomp_anom']; vars = ['vcompvcomp_anom']
        vars = ['qheating']
        vars = ['height']
        vars = ['stream']
        vars = ['flux_lhe']
        var = 'stream'
        var = 'height'
        var = 'olr'
        var = 'precipitation'
        var = 'ucomp'
        plotting_type = 'lat_lon_543'
        plotting_type = 'lat_lon_886'

    exp_control = 'gorman_tracmip_control_rerun' # Re-run has all levels
    exp_perturbs=['15NA100L', '30NA100L', '45NA100L', '60NA100L', '75NA100L']

    season = 'extended_summer'; season = 'extended_winter'
    season = 'annual'
    season = None #None probably means annual

    # Figure 2
    var='t_surf'
    plotting_type = 'lat_lon_surf' # Only lat-lon map without height selction goes into this
    resolution = 'monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090') # All-monthly up to here
    # Figure 4
    var='omega'
    plotting_type = 'lat_lon_989' #=990 hPa
    resolution = 'monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090') # All-monthly up to here
    # Figure 5
    var='ps'
    plotting_type = 'lat_lon_surf' 
    resolution = 'monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090') # All-monthly up to here
    # Figure 8
    var = 'vcomptheta_anom'
    plotting_type = 'lat_lon_989' #=990 hPa
    resolution = 'climatology'; control_years, perturb_years = None, None
    # Figure S4
    var = 'vor'
    plotting_type = 'lat_lon_232'
    resolution='monthly'; control_years=('0021','0050'); perturb_years=('0061', '0090') # All-monthly up to here
    # Figure S5
    var = 'EGR'
    plotting_type = 'lat_lon_surf' 
    #resolution = 'climatology'; control_years, perturb_years = None, None
    resolution = 'daily'; control_years = ('0021','0050'); perturb_years = ('0061', '0090')
    # Figure 3
    var='temp'
    plotting_type = 'height_lon' # dimension is longitude
    resolution = 'monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090') # All-monthly up to here
    # Revision figure
    var='temp'
    plotting_type='height_lat' # x-dimension is latitude
    resolution='monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090') # All-monthly up to here
    # Revision figure (new Figure S4 added)
    var='vcomptemp'
    plotting_type='height_lat' # x-dimension is latitude
    resolution='monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090') # All-monthly up to here

    if plotting_type=='height_lon': # Average across latitudes
        dim_value={'15NA100L':slice(0,30),'30NA100L':slice(15,45),'45NA100L':slice(30,60),'60NA100L':slice(45,75),
                '75NA100L':slice(60,90)}
        dimension='longitude'
    if plotting_type=='height_lat': # Average across longitudes of the heating perturbations
        dim_value={'15NA100L':slice(-16,16),'30NA100L':slice(-18,18),'45NA100L':slice(-22,22),'60NA100L':slice(-31,31),
                '75NA100L':slice(-60,60)}
        dimension='latitude'
    # Folloowing are lat lon at different levels
    elif plotting_type=='lat_lon_surf':
        dim_value={'15NA100L':None,'30NA100L':None,'45NA100L':None,'60NA100L':None,'75NA100L':None}
        dimension=None
    elif plotting_type=='lat_lon_989':
        plotting_type='lat_lon'
        dim_value={'15NA100L':slice(988,990),'30NA100L':slice(988,990),'45NA100L':slice(988,990),
                '60NA100L':slice(988,990),'75NA100L':slice(988,990)}
        dimension=None
    elif plotting_type=='lat_lon_543':
        plotting_type='lat_lon'
        dim_value={'15NA100L':slice(543,544),'30NA100L':slice(543,544),'45NA100L':slice(543,544),
                '60NA100L':slice(543,544),'75NA100L':slice(543,544)}
        dimension=None
    elif plotting_type=='lat_lon_232':
        plotting_type='lat_lon'
        dim_value={'15NA100L':slice(232,233),'30NA100L':slice(232,233),'45NA100L':slice(232,233),
                '60NA100L':slice(232,233),'75NA100L':slice(232,233)}
        dimension=None
    elif plotting_type=='lat_lon_886':
        plotting_type='lat_lon'
        dim_value={'15NA100L':slice(886,887),'30NA100L':slice(886,887),'45NA100L':slice(996,887),
                '60NA100L':slice(886,887),'75NA100L':slice(886,887)}
        dimension=None

    control_sel = {exp:{} for exp in exp_perturbs}
    response_sel = {exp:{} for exp in exp_perturbs}
    control=red.reading_vards(exp_control,var,years=control_years,season=season, lats=None,resolution=resolution,
                            height_lev=None, change_lons_order=True)
    for exp in exp_perturbs:
        if True: # Full response
            perturb=red.reading_vards(exp,var,years=perturb_years,season=season, lats=None,
                    resolution=resolution, height_lev=None, change_lons_order=True)
            if 'time' in control.dims:
                response = perturb.mean(dim='time') - control.mean(dim='time')
            else: # there is no time dimension in climatology
                response = perturb - control
            response_sel[exp] = red.select_dim(response, plotting_type=plotting_type, dim_value=dim_value[exp])
        else: # Derive the eddy response only using the peturb setup (This is only for lat-lon map)
            perturb = red.reading_vards(exp,var,years=perturb_years,season=season,lats=None,
                    resolution=resolution, height_lev=None, change_lons_order=False)
            response_sel[exp] = red.select_dim(perturb, plotting_type=plotting_type, dim_value=dim_value[exp])
            response_sel[exp] = response_sel[exp] - response_sel[exp].mean(dim='longitude')

    ### Set up the plotting grids
    shading_grids = [response_sel[exp] for exp in exp_perturbs]
    response_clevel = {'qheating':1, 'vor':4e-6, 'vcomptheta_anom':5, 'height':30, 'sphum':0.0005, 'temp':2, 
            'flux_lhe':100, 'stream':2e6, 'ps':400}
    response_clevel = {'qheating':1, 'vor':6e-6, 'vcomptheta_anom':8, 'ucompvcomp_anom':4, 'ucompucomp_anom':5,
            'vcompvcomp_anom':5, 'height':30, 'sphum':0.0005,'olr':10,
            'temp':3, 'flux_lhe':100, 'ucomp':3, 'vcomp':1, 'stream':2e6, 'ps':300, 'omega':0.5e-2, 't_surf':5, 
            'precipitation':5e-5, 'EGR':0.3, 'vcomptemp':500}
    shading_clevels = [np.linspace(response_clevel[var]*-1, response_clevel[var], 11)] * len(shading_grids)
    #shading_clevels = [np.linspace(response_clevel[var]*-1, response_clevel[var], 21)] * len(shading_grids)

    if False: # Plot the control setup as contour
        contour_map_grids = [control_sel[var] for exp in exp_perturbs]
        control_clevel = {'ucomp':30,'temp':500, 'omega':21, 'omegatheta_anom':0.15,'qheating':1,'precipitation':11,
                'precipitation':11, 'vor':1e-5, 'vcomptheta_anom':15, 'sphum':0.02, 'height':500, 'flux_lhe':200}
        contour_clevels = [np.linspace(control_clevel[var]*-1, control_clevel[var], 11)] * len(shading_grids)
    else: # Contour follows shading
        contour_map_grids = shading_grids
        contour_clevels = shading_clevels
        #contour_map_grids = None # Not contours
        #contour_clevels = None

    ### Start the plotting 
    cols=len(exp_perturbs); rows=1
    grid_no = cols * rows
    if False:
        cmap = 'coolwarm'
    else:
        mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff', '#ffffff', '#fddbc7',  '#f4a582', '#d6604d', '#b2182b']
        cmap = matplotlib.colors.ListedColormap(mapcolors)
    cmap_grids= [cmap] * grid_no
    clabels_rows = [''] * grid_no
    left_titles = [''] * rows; top_titles = [''] * cols
    ABC = ['A', 'B', 'C', 'D', 'E', 'F']
    ind_titles = ['('+ABC[i]+') '+exp[0:3] for i, exp in enumerate(exp_perturbs)]

    xsize=2; ysize=2
    if 'height_lat'==plotting_type:
        xticklabels = [-90,-60,-30,0,30,60,90]
        xticklabels = [-60,-30,0,30,60,90]
        yticklabels = [10,200,400,600,800,1000]
        ut.grid_plotting_height_lat(shading_grids, rows, cols, cmap_grids, shading_clevels, clabels_rows,
            top_titles=top_titles, grid=False, left_titles=left_titles, ind_titles=ind_titles, log_height=False,
            xticklabels=xticklabels, yticklabels=yticklabels, dimension=dimension,
            contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, 
            transpose=False,xsize=xsize,ysize=ysize,xlabel='latitude',
            contour_label=False)
    elif 'height_lon'==plotting_type:
        xticklabels = [-120, -60, 0, 60, 120] # set change_lons_order=True
        yticklabels = [10,200,400,600,800,1000]
        ut.grid_plotting_height_lat(shading_grids, rows, cols, cmap_grids, shading_clevels, clabels_rows,
            top_titles=top_titles, grid=False,left_titles=left_titles, ind_titles=ind_titles, log_height=False,
            xticklabels=xticklabels, yticklabels=yticklabels, dimension=dimension,contour_map_grids=contour_map_grids,
            contour_clevels=contour_clevels, transpose=False, xsize=xsize,ysize=ysize, xlabel='longitude')
    ###############
    if 'lat_lon' in plotting_type:
        freetext = None; freetext_pos = None
        pval_grids = None; pval_hatches = None
        lat_ticks = [-60,-30,0,30,60]
        lon_ticks = [-180,-120,-60,0,60,120,180]
        projection=ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90,globe=None)
        xsize=2; ysize=2; low_lat=0; gridline=True
        map_extents = {'left':-50, 'right':50, 'bottom':60, 'top':90} 
        map_extents=None
        if exp_perturb=='75NA100_qflux':
            lat1=82.5; lat2=67.5; lon1=-45; lon2=45
        elif exp_perturb=='60NA100_qflux':
            lat1=67.5; lat2=52.5; lon1=-45; lon2=45
        elif exp_perturb in ['45NA37_qflux', '45NA100_qflux']:
            lat1=52.5; lat2=37.5; lon1=-45; lon2=45
        elif exp_perturb == '15NA100_qflux':
            lat1=22.5; lat2=7.5; lon1=-45; lon2=45
        #region_boxes = [ut.create_region_boxes(lat1, lat2, lon1, lon2)] * grid_no
        region_boxes = None
        plotting_crosses=[15,30,45,60,75]
        ut.map_grid_plotting(shading_grids, rows, cols, cmap_grids, shading_clevels, clabels_rows,
        top_titles=top_titles,left_titles=left_titles, projection=projection, low_lat=low_lat,
        coastlines=False, xsize=xsize,ysize=ysize, gridline=True, ind_titles=ind_titles,
        region_boxes=region_boxes, map_extents=map_extents,contour_map_grids=contour_map_grids,
        contour_clevels=contour_clevels, shading_extend='both',
        freetext=freetext, freetext_pos=freetext_pos, pval_hatches=pval_hatches, pval_map=pval_grids,
        circle=True, transpose=False, quiver_grids=None, contour_label=False, plotting_crosses=plotting_crosses)
    fname='combine_plotting'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=500, pad_inches=0.01)

def plotting_U850_annual_timeseries():

    # Plotting the zonal-mean U850 versus latitude (To compare with Voigt et al. 2016)

    exp = 'gorman_tracmip'
    var= 'ucomp' # var name for model

    vards = reading_vards(exp , var)
    # Extract the last 10 years
    vards = vards.sel(time=slice('0010-01-01', '0020-12-15'))

    # Extract 850 hpa and plot the timeseries
    vards_extract = extract_period(vards, 'annual')
    vards_mean = vards_extract.mean(dim='longitude').mean(dim='time') # have a dimension of latitude and height
    # Do interpolation
    vards_mean_850 = vards_mean.interp(lev_hPa=850)

    times = [vards_mean_850.latitude]
    timeseries = [vards_mean_850]
    ylabel = 'Zonal wind (m/s)'
    xlabel = 'Latitude'
    fname = 'zonal_wind_850hPa'
    xticks=[-60,-30,0,30,60]
    ylims=[-10,15]
    ut.timeseries_plotting(times, timeseries, ylabel, xlabel=xlabel, fname=fname, grid=True, xticks=xticks, ylims=ylims)

def zonal_wind_snap_shot():

	# Make the grids. For the first 10 months
	map_grids = [vards_var.isel(time=i) for i in range(0, len(runs))]
	leftcorner_text = ['run:'+str(i+1) for i in runs]
	rows = 5
	cols = 5

	mapcolors, _ = ut.map_raw_anom(raw_colors=8)
	cmap = matplotlib.colors.ListedColormap(mapcolors)
	cmap = 'Blues'
	clevels_rows = [np.linspace(97000,102000,9)] * len(map_grids)
	clevels_rows = [np.linspace(-16,16,9)] * len(map_grids)
	clabels_row = ['Pa'] * rows
	left_titles = [''] * rows
	top_titles = [''] * cols
	projection = ccrs.PlateCarree()

	# Do the plotting
	ut.map_grid_plotting(map_grids,rows,cols,cmap,clevels_rows,clabels_row, top_titles=top_titles, left_titles=left_titles, 
						subplot_adjust=True, projection=projection,xsize=4,ysize=2, low_lat=-90, coastlines=False, gridline=False, leftcorner_text=leftcorner_text)
	fname = 'SLP_every_month'
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
	plt.savefig('../graphs/%s_%s.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=150)

