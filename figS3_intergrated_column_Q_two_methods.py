import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
from importlib import reload

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')
import read_extract_data as red
import utilities as ut
import figS3_intergrated_column_Q_two_methods as pcQ


if __name__ == "__main__":

    control = 'gorman_tracmip_control_rerun'; perturb = '75NA100_qflux_rerun'
    control = 'gorman_tracmip_control'; perturb = '75NA100_qflux'
    control = 'gorman_tracmip_control'; perturb = '15NA100W12_qflux'
    control = 'gorman_tracmip_control_rerun'; perturb = '75NA100L'
    control = 'gorman_tracmip_control_rerun'; perturb = '60NA100L'
    years = {control:('0021', '0050'), perturb:('0061','0090')}

    exps=['gorman_tracmip_control_rerun','15NA100L','30NA100L','45NA100L','60NA100L','75NA100L']
    years=[('0021','0050')] + [('0061','0090')]*5

    Q_residual = {}
    Q_boundary = {}
    for i, exp in enumerate(exps):
        Ras, shfs, LPs = pcQ.get_column_diabatic_heating_component(exp, years[i])
        Q_boundary[exp] = Ras + shfs + LPs
        Q_residual[exp] = pcQ.getting_Q_residual_intergral(exp)

    exps=['15NA100L','30NA100L','45NA100L','60NA100L','75NA100L']
    shading_grids=[]
    for exp in exps:
        Q_boundary_response=Q_boundary[exp]-Q_boundary['gorman_tracmip_control_rerun']
        shading_grids.append(Q_boundary_response)
    for exp in exps:
        Q_residual_response=Q_residual[exp]-Q_residual['gorman_tracmip_control_rerun']
        shading_grids.append(Q_residual_response)


    left_titles = ['Boundary\nmethod', 'Residual\nmethod']
    ABC = ['A', 'B', 'C', 'D', 'E', 'F']
    top_titles = ['('+ABC[i]+') '+exp[0:3] for i, exp in enumerate(exps)]
    ind_titles = None
    #from qflux import create_qflux_input_file as cqif
    #cqif.check_global_intergral(Q[perturb]-Q[control], Q[control].longitude.values, Q[control].latitude.values)

    # Contour with shading
    contour_grids = shading_grids
    rows = 2; cols=len(exps)
    grids = rows*cols
    shading_clevels = [np.linspace(-90,90,13)] * grids
    contour_clevels = shading_clevels
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#eef7fa', '#ffffff', '#ffffff', 
                '#fff6e5', '#fddbc7',  '#f4a582', '#d6604d', '#b2182b']
    cmap = matplotlib.colors.ListedColormap(mapcolors)
    mapcolors_grids = [cmap] * grids
    clabels_rows = [''] * rows
    freetext = None; freetext_pos = None
    pval_map = None; pval_hatches = None
    lat_ticks = None
    lon_ticks = None
    #projection = ccrs.PlateCarree(); xsize=3; ysize=2; low_lat=-90; gridline=False
    projection=ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90,globe=None)
    xsize=2; ysize=1.5; low_lat=0; gridline=True
    map_extents=None
    map_extents = {'left':-60, 'right':60, 'bottom':0, 'top':90}
    plotting_crosses=[15,30,45,60,75] * 2
    plotting_crosses=None
    ut.map_grid_plotting(shading_grids, rows, cols, mapcolors_grids, shading_clevels, clabels_rows, top_titles=top_titles,
            left_titles=left_titles,projection=projection,low_lat=low_lat, coastlines=False, xsize=xsize, ysize=ysize,
            gridline=gridline,ind_titles=ind_titles, region_boxes=None, map_extents=map_extents,
            contour_map_grids=contour_grids, plotting_crosses=plotting_crosses,
            contour_clevels=contour_clevels, shading_extend='both', pval_hatches=pval_hatches,
            pval_map=pval_map, circle=False, lat_ticks=lat_ticks, lon_ticks=lon_ticks, cb_individual=True, contour_label=False)
    fname = 'Q_intergral'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.4)
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=400, pad_inches=0.01)


def getting_Q_residual_intergral(exp):

	# This is the intergrated Q using the residual method

	var = "qheating"
	resolution = 'climatology'
	control_years=None
	season = None
	control_Q = red.reading_vards(exp, var, years=control_years, season=season, lats=None, resolution=resolution, height_lev=None, change_lons_order=True)

	if True: ### Interpolate the 1000hPa and 0hPa data (by simply copying data from the nearest level)
		control_Q_new = np.zeros((control_Q.shape[0]+2, control_Q.shape[1], control_Q.shape[2]))
		control_Q_new[1:-1] = control_Q
		control_Q_new[0] = control_Q[0]
		control_Q_new[-1] = control_Q[-1]
		new_lev_dim = np.zeros((control_Q.shape[0]+2))
		new_lev_dim[1:-1] = control_Q.lev_hPa.values
		new_lev_dim[0] = 0
		new_lev_dim[-1] = 1000
		control_Q.lev_hPa.values
		control_Q_new = xr.DataArray(control_Q_new, dims=['lev_hPa', 'latitude', 'longitude'],
						coords={'lev_hPa':new_lev_dim, 'latitude':control_Q.latitude, 'longitude':control_Q.longitude})
		control_Q = control_Q_new 

	Cp=718 # Actually this is Cv
	Cp=1004 # J/K/kg
	g = 9.81
	hPa_to_Pa = 100
    # Q is unit of K/day
    # Pa is the unit of N/m^2
    # Cp has unit of J/K/kg
    # g has unit of m/s^2
    # The output unit is Wm-2
	Q_intergral = control_Q.integrate('lev_hPa') * hPa_to_Pa * Cp / (24*3600) / g

	return Q_intergral


def get_column_diabatic_heating_component(exp, years):

    ### Boundary method
	### Calculate the Q by the method of Trenberth, Amy Solomon 1993

	resolution = 'monthly'
	height_lev = None
	lats = None
	season=None
	vars = ['flux_lhe', 'flux_t', 'lwup_sfc', 'lwdn_sfc', 'swdn_toa', 'swdn_sfc', 'olr', 'precipitation', 'convection_rain', 'condensation_rain']
	vars = ['lwup_sfc', 'lwdn_sfc', 'swdn_sfc', 'swdn_toa', 'olr', 'flux_t', 'precipitation']
	flux_mean = {}
	for var in vars:
		flux = red.reading_vards(exp,var,years=years,season=season, lats=lats, resolution=resolution, height_lev=height_lev, change_lons_order=True)
		flux_mean[var]= flux.mean(dim='time')
	albedo = 0.38
	absorbed_solar_surface = flux_mean['swdn_sfc']
	# This downward surface solar flux is constant in different albedo. How the atmosphere absorbs solar radiation is prescribed
	downward_solar_surface = absorbed_solar_surface / (1-albedo) # Downward flux*(1-albedo)=absorbed flux.
	upward_solar_surface = downward_solar_surface * albedo
	thermals = flux_mean['flux_t']
	incoming_solar = flux_mean['swdn_toa']
	outgoing_longwave_toa = flux_mean['olr']
	upward_longwave_surface = flux_mean['lwup_sfc']
	downward_longwave_surface = flux_mean['lwdn_sfc']
	#absorbed_solar = incoming_solar - absorbed_solar_surface - upward_solar_surface # This seems to be the absorbed solar radiation by atmosphere
	absorbed_solar = incoming_solar-downward_solar_surface # This is the absorbed solar radiation by the atmosphere (the same across different perturbation exps since albedo is the same)
	surface_longwave = upward_longwave_surface - downward_longwave_surface
    # Here we take downward flux as positive and upward flux as negative.
    #Ra (net radiation) = Rt - Rs # (according to Hartmann textbook Equation 6.2)
	Rt = incoming_solar - upward_solar_surface - outgoing_longwave_toa
	Rs = downward_solar_surface - upward_solar_surface + downward_longwave_surface - upward_longwave_surface
	net_radiation = absorbed_solar + surface_longwave - outgoing_longwave_toa # This is the same as Rt-Rs
	#LP = (flux_mean['convection_rain'] + flux_mean['condensation_rain']) * 2.26e6
	LP = flux_mean['precipitation'] * 2.26e6
	#Q = net_radiation + thermals + LP

	return net_radiation, thermals, LP

def backup():
    ### Plot the Q_intergral
    if False: # By energy budget at the boundary surface
        #shading_grids = (Ras[perturb]-Ras[control], shfs[perturb]-shfs[control], LPs[perturb]-LPs[control], Q_boundary[perturb]-Q_boundary[control])
        #ind_titles = ['Net radiation', 'SHF', 'LP', 'Total Q']
        #shading_clevels = [np.linspace(-20,20,11)] + [np.linspace(-50,50,11)]*2 + [np.linspace(-300,300,11)]
        shading_grids = [Q_boundary[perturb]-Q_boundary[control]]
        ind_titles = ['Q by boundary method']
        shading_clevels = [np.linspace(-60,60,11)]
    if False:  # By the reisudal method
        shading_grids = [Q_residual[perturb]- Q_residual[control]]
        ind_titles = ['Q by residual']
        shading_clevels = [np.linspace(-300,300,11)] 
