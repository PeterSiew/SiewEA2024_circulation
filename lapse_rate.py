import numpy as np
import xarray as xr
import ipdb
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
from importlib import reload

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')

from qflux import create_qflux_input_file as cqif
import plot_column_Q as pcQ
import read_extract_data as red
import utilities as ut

if __name__ == "__main__":

    # new experiments
    exp_control= 'gorman_tracmip_control_rerun'
    exp_perturb = ['75NA100L', '60NA100L', '45NA100L', '30NA100L', '15NA100L']
    exp_perturb_name = ['75N', '60N', '45N', '30N', '15N']
    exps = exp_perturb + [exp_control]
    years = [['0061','0090']]*len(exp_perturb) + [['0021', '0050']]

    # Define the regions of the heating perturbations
    qflux_latitudes = {'75NA100_qflux':(67.5,82.5), '45NA100W16_qflux':(37.5,52.5), '15NA100W12_qflux':(7.5,22.5),
            '75NA100L':(60,90), '60NA100L':(45,75), '45NA100L':(30,60), '30NA100L':(15,45), '15NA100L':(0,30)}
    qflux_longitudes = {'75NA100_qflux':(-45,45), '45NA100W16_qflux':(-16,16), '15NA100W12_qflux':(-12,12),
            '75NA100L':(-60,60),'60NA100L':(-31.1,31.1),'45NA100L':(-21.95,21.95),'30NA100L':(-17.95,17.95),'15NA100L':(-16.1,16.1)}

    ### Get the time-mean of these variables
    var = 'temp'
    height_lev = None
    lats = None
    season=None
    resolution = 'monthly'
    flux_mean = {exp:{} for exp in exps}
    for i, exp in enumerate(exps):
        flux = red.reading_vards(exp,var,years=years[i],season=season, lats=lats, resolution=resolution,
                height_lev=height_lev, change_lons_order=True)
        flux_mean[exp]=flux.mean(dim='time').compute()


    ### Average over the box
    temp_control = flux_mean[exp_control]
    perturb_temp= {}
    control_temp = {}
    for exp in exp_perturb:
        print(exp)
        temp_perturb = flux_mean[exp]
        lat1=qflux_latitudes[exp][0]; lat2=qflux_latitudes[exp][1]
        lon1=qflux_longitudes[exp][0]; lon2=qflux_longitudes[exp][1]
        perturb_temp[exp]=temp_perturb.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2)).mean(dim='latitude').mean(dim='longitude')
        control_temp[exp]=temp_control.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2)).mean(dim='latitude').mean(dim='longitude')

    # Plot the control lapse rate and the perturb lapse rates
    plt.close()
    fig, axs = plt.subplots(1,len(exp_perturb),figsize=(7,2))
    for i, exp in enumerate(exp_perturb):
        axs[i].plot(control_temp[exp][::-1], control_temp[exp].lev_hPa.values[::-1], color='black', lw=0.5)
        axs[i].plot(perturb_temp[exp][::-1], perturb_temp[exp].lev_hPa.values[::-1], color='red', lw=0.5)
        axs[i].invert_yaxis()
        axs[i].set_title(exp[0:3])
    for ax in axs[1:]:
        ax.tick_params(axis='y', which='both',length=2)
        ax.set_yticks([])
    for ax in axs:
        ax.set_xticks([250,300])
        ax.set_xlim([200,320])
        ax.set_ylim([1000,100])
        #ax.set_xticklabels([200,250])
    axs[0].set_ylabel('Pressure')
    axs[2].set_xlabel('Temperature (K)')
    fname='lapse_rate'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.2)
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=300, pad_inches=0.1)





