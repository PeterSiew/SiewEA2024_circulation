import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import math
from importlib import reload

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')
import read_extract_data as red
import utilities as ut
import heat_budget as hb

import read_extract_data as red

if __name__ == "__main__":

    # Here we need to read all the raw data


    exps=['gorman_tracmip_spinup', 'gorman_tracmip_control','15NA100L', '30NA100L', '45NA100L', '60NA100L', '75NA100L']
    var = "t_surf"
    #resolution = 'monthly'; control_years = ('0021','0050'); perturb_years = ('0061', '0090')
    resolution = 'monthly'
    #years_control= {'gorman_tracmip_control_rerun':('0001','0060')}
    #years_perturb={'15NA100L':('0051','0090'), '30NA100L':('0051','0090'), '45NA100L':('0051','0090'), '60NA100L':('0051','0090'), '75NA100L':('0051','0090')}
    #years = {**years_control, **years_perturb}
    runs={'gorman_tracmip_spinup':240,'gorman_tracmip_control':360,'15NA100L':480, '30NA100L':480, '45NA100L':480, '60NA100L':480, '75NA100L':480}
    season=None
    resolution='monthly'

    theta_ts_annual={}
    for exp in exps:
        thetas=[]
        for run in range(1,runs[exp]+1):
            run=str(run).zfill(4)
            theta = xr.open_dataset('/data0/pyfsiew/raw_output/%s/run%s/atmos_monthly.nc'%(exp,run),chunks={'time':100})
            thetas.append(theta[var])
        thetas = xr.concat(thetas,dim='time').compute()
        theta_ts = thetas.mean(dim='lon').mean(dim='lat') # Global SST
        theta_ts_annual[exp] = theta_ts.groupby(theta_ts.time.dt.year).mean(dim='time')
        # Extract the timeseries at lonigutde=0, lat=75
        #theta_ts = theta.sel(longitude=slice(0,90)).sel(latitude=slice(70,80)).mean(dim='longitude').mean(dim='latitude')
        #theta = red.reading_vards(exp, var, years=years[exp], season=season, lats=None, resolution=resolution, height_lev=None)

    colors={'gorman_tracmip_spinup':'gray','gorman_tracmip_control':'k','15NA100L':'palegoldenrod', '30NA100L':'orange', '45NA100L':'coral', '60NA100L':'orangered', '75NA100L':'brown'}
    labels={'gorman_tracmip_spinup':'Spin-up','gorman_tracmip_control':'Control','15NA100L':'15N', '30NA100L':'30N', '45NA100L':'45N', '60NA100L':'60N', '75NA100L':'75N'}
    # Plot the timeseries
    fig, ax1 = plt.subplots(1,1,figsize=(6,3))
    for exp in exps:
        x = theta_ts_annual[exp].year
        ax1.plot(x, theta_ts_annual[exp], linestyle='-', color=colors[exp], label=labels[exp])
    xticks=[1,10,20,30,40,50,60,70,80,90]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)
    ax1.set_ylabel('Global sea surface temperature (K)')
    ax1.set_xlabel('Years (Annual-mean)')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax1.legend()
    fname = 'SST_evolution_control_and_peturb'
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=400)

