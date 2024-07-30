import numpy as np
import xarray as xr
import ipdb
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
from importlib import reload

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')

from qflux import create_qflux_input_file as cqif
import read_extract_data as red
import utilities as ut

if __name__ == "__main__":

    reload(cqif)

    ##################################################

    # Old one - ready to delect
    exp_control='gorman_tracmip_control'; year_control = ('0021', '0050')
    exp_perturb ='45NA100W16_qflux'; year_perturb = ('0061', '0090')
    exp_perturb ='15NA100W12_qflux'; year_perturb = ('0061', '0090')
    exp_perturb ='75NA100_qflux'; year_perturb = ('0061', '0090')
    # new experiments
    exp_control= 'gorman_tracmip_control_rerun'
    exp_perturb = ['75NA100L', '60NA100L', '45NA100L', '30NA100L', '15NA100L'][::-1]
    exp_perturb_name = ['75N', '60N', '45N', '30N', '15N'][::-1]
    exps = exp_perturb + [exp_control]
    years = [['0061','0090']]*len(exp_perturb) + [['0021', '0050']]

    # Define the regions of the heating perturbations
    qflux_latitudes = {'75NA100_qflux':(67.5,82.5), '45NA100W16_qflux':(37.5,52.5), '15NA100W12_qflux':(7.5,22.5),
            '75NA100L':(60,90), '60NA100L':(45,75), '45NA100L':(30,60), '30NA100L':(15,45), '15NA100L':(0,30)}
    qflux_longitudes = {'75NA100_qflux':(-45,45), '45NA100W16_qflux':(-16,16), '15NA100W12_qflux':(-12,12),
        '75NA100L':(-60,60),'60NA100L':(-31.1,31.1),'45NA100L':(-21.95,21.95),'30NA100L':(-17.95,17.95),'15NA100L':(-16.1,16.1)}

    ### Get the time-mean of these variables
    vars = ['lwup_sfc', 'lwdn_sfc', 'olr', 'flux_t', 'precipitation', 'flux_lhe', 't_surf', 'temp']
    height_lev = None
    lats = None
    season=None
    resolution = 'monthly'
    flux_mean = {exp:{} for exp in exps}
    for i, exp in enumerate(exps):
        for var in vars:
            flux = red.reading_vards(exp,var,years=years[i],season=season, lats=lats, resolution=resolution,
                    height_lev=height_lev, change_lons_order=True)
            #if var=='precipitation':
            #    flux = flux*2.26e6
            if var=='temp':
                #flux=flux.sel(lev_hPa=slice(989.2,989.4)).mean(dim='lev_hPa') # only surface
                flux=flux.mean(dim='lev_hPa') # Average across the depth
            if True: # Do winter and summer
                #mon_mask=flux.time.dt.month.isin([12,1,2])
                mon_mask=flux.time.dt.month.isin([6,7,8])
                flux=flux.sel(time=mon_mask)
            flux_mean[exp][var]=flux.mean(dim='time')

    ### Get the residual (assume this is the MSE circulation)
    for exp in exps:
        flux_mean[exp]['circulation'] = flux_mean[exp]['lwup_sfc'] - flux_mean[exp]['lwdn_sfc'] + \
                                        flux_mean[exp]['flux_t'] + flux_mean[exp]['flux_lhe'] - flux_mean[exp]['olr']
        flux_mean[exp]['lwnet_sfc'] = flux_mean[exp]['lwup_sfc'] - flux_mean[exp]['lwdn_sfc']

    if False:  ### Get the control and perturb air temperature and OLR
        temp_mean_perturb={exp:{} for exp in exp_perturb}
        temp_mean_control={exp:{} for exp in exp_perturb}
        area = 11172375725010 # Using 75NA100L heating size as a reference. Others are more or less the same
        for exp in exp_perturb:
            for var in ['t_surf', 'olr']:
                data=flux_mean[exp][var]
                lat_mask = (data.latitude>=qflux_latitudes[exp][0]) & (data.latitude<=qflux_latitudes[exp][1]) 
                lon_mask = (data.longitude>=qflux_longitudes[exp][0]) & (data.longitude<=qflux_longitudes[exp][1]) 
                heating_mask = (lat_mask & lon_mask) # select the heating region
                ### This weight calculation is consistent with the following 
                control_flux = xr.where(~heating_mask, 0, flux_mean[exp_control][var]) 
                temp_mean_control[exp][var]= cqif.check_global_intergral(control_flux.values, control_flux.longitude.values,
                                                                            control_flux.latitude.values)/area 
                perturb_flux = xr.where(~heating_mask, 0, flux_mean[exp][var]) 
                temp_mean_perturb[exp][var]= cqif.check_global_intergral(perturb_flux.values, perturb_flux.longitude.values,
                                                                            perturb_flux.latitude.values)/area 
                ### Don't do the mean
                #temp_mean_control[exp][var]=flux_mean[exp_control][var].values[heating_mask].mean()
                #temp_mean_perturb[exp][var]=flux_mean[exp][var].values[heating_mask].mean()
        for exp in exp_perturb:
            emi=0.76; sigma=5.67e-18
            temp1=temp_mean_control[exp]['t_surf']
            temp2=temp_mean_perturb[exp]['t_surf']
            olr1=temp_mean_control[exp]['olr']
            olr2=temp_mean_perturb[exp]['olr']
            temp_increase=temp2-temp1
            olr_increase=olr2-olr1
            feedback=olr_increase/temp_increase
            predicted_olr_increase=((emi*sigma*temp2**4)-(emi*sigma*temp1**4)) / (emi*sigma*temp1**4) *100
            #print(exp,temp_increase,olr_increase,feedback,predicted_olr_increase)
            print(exp,temp_increase,olr_increase,feedback)
                                    
    ### Get the difference between peruturb and control
    vars = ['flux_t', 'flux_lhe', 'lwnet_sfc', 'olr', 'circulation', 't_surf', 'temp']
    flux_diff = {exp:{} for exp in exp_perturb}
    for exp in exp_perturb:
        for var in vars:
            flux_diff[exp][var] = flux_mean[exp][var] - flux_mean[exp_control][var]

    ### Average over the box
    netflux_sum = {exp:{} for exp in exp_perturb}
    area = 11172375725010 # Using 75NA100L heating size as a reference. Others are more or less the same
    for exp in exp_perturb:
        #print(exp)
        for var in vars:
            netflux = flux_diff[exp][var]
            lat_mask = (netflux.latitude>=qflux_latitudes[exp][0]) & (netflux.latitude<=qflux_latitudes[exp][1]) 
            lon_mask = (netflux.longitude>=qflux_longitudes[exp][0]) & (netflux.longitude<=qflux_longitudes[exp][1]) 
            heating_mask = (lat_mask & lon_mask) # select the heating region
            netflux_heating_region = xr.where(~heating_mask, 0, netflux) 
            #netflux_noheating_region = xr.where(heating_mask, 0, netflux) 
            netflux_sum[exp][var]= cqif.check_global_intergral(netflux_heating_region.values,
                    netflux_heating_region.longitude.values, netflux_heating_region.latitude.values)/area # Do /1e12 for T2
            #netflux_sum[exp][var] = netflux_heating_region.values[heating_mask].mean() # THis is not the weight average
            if var=='t_surf': # Should do the area-weight
                netflux_sum[exp][var] = netflux_heating_region.values[heating_mask].mean()

    if True: # Plot the energy budget in a bar chart
        shfs = np.array([netflux_sum[exp]['flux_t'] for exp in exp_perturb])
        lhfs = np.array([netflux_sum[exp]['flux_lhe'] for exp in exp_perturb])
        netlw = np.array([netflux_sum[exp]['lwnet_sfc'] for exp in exp_perturb])
        #print(lhfs/(shfs+lhfs+netlw)*100) # the ratio between lhfs and all other input fluxes
        olr = np.array([netflux_sum[exp]['olr'] for exp in exp_perturb])
        circu = np.array([netflux_sum[exp]['circulation'] for exp in exp_perturb])
        x = np.arange(len(exp_perturb))
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,2))
        bw=0.2; xoff=0.1
        ax1.bar(x-xoff, shfs, bw, linestyle='-', bottom=0, color='red', label='Surface sensible')
        ax1.bar(x-xoff, lhfs, bw, linestyle='-', bottom=shfs, color='orange', label='Surface latent')
        ax1.bar(x-xoff, netlw, bw, linestyle='-', bottom=shfs+lhfs, color='gold', label='Surface net longwave')
        ax1.bar(x+xoff, olr, bw, linestyle='-', bottom=0, color='royalblue', label='Outgoing longwave')
        ax1.bar(x+xoff, circu, bw, linestyle='-', bottom=olr, color='lightblue', label='Circulation (residual)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_perturb_name)
        #ax1.set_ylabel('Energy (TW)')
        ax1.set_ylabel('Anomalous energy \n (Wm-2)')
        ax1.set_xlabel('Heating experiment')
        if False: # Plot the surface temperature
            ax2 = ax1.twinx()
            #t990= np.array([netflux_sum[exp]['T990'] for exp in exp_perturb])
            #ax2.plot(x,t990,'green', label='990hPa temp')
            surft = np.array([netflux_sum[exp]['t_surf'] for exp in exp_perturb])
            ax2.plot(x,surft,'k', label='Surface temp')
            ax2.legend(bbox_to_anchor=(0.9, -0.4), ncol=1, loc='lower left', frameon=False,
                            columnspacing=1, handletextpad=0.2, labelspacing=0.1)
            ax2.set_ylabel('Anomalous\ntemperature (K)')
            ax2.set_yticks([1,3,5])
            ax_all = [ax1,ax2]
        for i in ['top', 'bottom', 'left', 'right']:
            for ax in [ax1]:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        ax1.legend(bbox_to_anchor=(-0.1, 0.9), ncol=2, loc='lower left',
                    frameon=False, columnspacing=0.5, handletextpad=0.1, labelspacing=0)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.2)
        fname='energy_budget_bar'
        plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=500, pad_inches=0.1)

def feedback_calculation():

    if False: # Plot the feedback processes (not used anymore)
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(4,2))
        olr = np.array([netflux_sum[exp]['olr'] for exp in exp_perturb])
        circu = np.array([netflux_sum[exp]['circulation'] for exp in exp_perturb])
        surft = np.array([netflux_sum[exp]['t_surf'] for exp in exp_perturb])
        circu_feedback = -circu / surft
        temp_feedback =  -olr / surft
        ###
        bw=0.5; xoff=0
        x = np.arange(len(exp_perturb))
        #ax1.bar(x-xoff, circu_feedback, bw, linestyle='-', bottom=0, color='lightblue', label='Circulation feedback')
        ax1.bar(x+xoff, temp_feedback, bw, linestyle='-', bottom=0, color='royalblue', label='Temperature feedback')
        ax1.axhline(y=0, color='k')
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_perturb_name)
        ax1.set_ylabel('Temperature feedbacks\n(W/m2/K)')
        #ax1.set_xlabel('Latitudes of the heating peturbation')
        fname='feedbacks'
        for i in ['top', 'bottom', 'left', 'right']:
            for ax in [ax1]:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
        #ax1.legend(bbox_to_anchor=(0.1, 0.2), ncol=1, loc='lower left',
        #            frameon=False, columnspacing=0.5, handletextpad=0.1, labelspacing=0)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.2)
        plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=300, pad_inches=0.1)

        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(4,2))
        #surft_cube = surft**3
        surft_cube = surft**4
        ax1.plot(surft_cube, olr)
        ax1.scatter(surft_cube, olr)
        fname='relationship_feedback_surft'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.2)
        plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=300, pad_inches=0.01)


def energy_budget_box_del():
    if False:
        # Plot the energy budget of a box
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(4,5))

        unit='TW'
        var_name='SHF'; var='flux_t'
        x=0; y_end=0.5; y_st=0.3; ytext_off=0.19
        ax1.annotate('', xy=(x, y_end), xytext=(x, y_st), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle= '->', color='black', lw=3.5, ls='-'))
        ax1.annotate("%s\n %s%s"%(var_name,str(round(netflux_sum[var],1)),unit), xy=(x, y_st-ytext_off), xytext=(x, y_st-ytext_off), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        var_name='LP'; var='precipitation'
        var_name='Latent'; var='flux_lhe'
        x=1; y_end=0.5; y_st=0.3; ytext_off=0.19
        ax1.annotate('', xy=(x, y_end), xytext=(x, y_st), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle= '->', color='black', lw=3.5, ls='-'))
        ax1.annotate("%s\n %s%s"%(var_name,str(round(netflux_sum[var],1)),unit), xy=(x, y_st-ytext_off), xytext=(x, y_st-ytext_off), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        var_name='Up LW'; var='lwup_sfc'
        x=2; y_end=0.5; y_st=0.3; ytext_off=0.19
        ax1.annotate('', xy=(x, y_end), xytext=(x, y_st), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle= '->', color='black', lw=3.5, ls='-'))
        ax1.annotate("%s\n %s%s"%(var_name,str(round(netflux_sum[var],1)),unit), xy=(x, y_st-ytext_off), xytext=(x, y_st-ytext_off), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        var_name='Down LW'; var='lwdn_sfc'
        x=3; y_end=0.5; y_st=0.3; ytext_off=0.19
        ax1.annotate('', xy=(x, y_end), xytext=(x, y_st), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle= '<-', color='black', lw=3.5, ls='-'))
        ax1.annotate("%s\n %s%s"%(var_name,str(round(netflux_sum[var],1)),unit), xy=(x, y_st-ytext_off), xytext=(x, y_st-ytext_off), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        var_name='OLR'; var='olr'
        x=0.5; y_end=1.2; y_st=1; ytext_off=0.02
        ax1.annotate('', xy=(x, y_end), xytext=(x, y_st), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle= '->', color='black', lw=3.5, ls='-'))
        ax1.annotate("%s\n %s%s"%(var_name,str(round(netflux_sum[var],1)),unit), xy=(x, y_end+ytext_off), xytext=(x, y_end+ytext_off), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        var_name='Circulation'; var='circulation'
        x=4; y_end=2; y_st=2; ytext_off=0.02
        ax1.annotate('', xy=(4, 0.6), xytext=(3, 0.6), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle= '->', color='black', lw=3.5, ls='-'))
        ax1.annotate("%s:\n %s%s"%(var_name,str(round(netflux_sum[var],1)),unit), xy=(3.5,0.65), xytext=(3.6,0.61), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        # Plot the t_surf
        var_name='Ts'; var='t_surf'
        ax1.annotate("%s: %sK"%(var_name,str(round(netflux_sum[var],1))), xy=(1,0.55), xytext=(1.5,0.51), xycoords='data',
                    textcoords='data', va = "bottom", ha="center")

        #energy_in = netflux_sum['flux_t'] + netflux_sum['lwup_sfc'] + netflux_sum['precipitation']
        #energy_out = netflux_sum['olr'] + netflux_sum['lwdn_sfc'] + netflux_sum['circulation']
        #energy_in = netflux_sum['flux_t'] + netflux_sum['lwup_sfc'] + netflux_sum['precipitation'] - netflux_sum['lwdn_sfc']
        energy_in = netflux_sum['flux_t'] + netflux_sum['lwup_sfc'] + netflux_sum['flux_lhe'] - netflux_sum['lwdn_sfc']
        energy_out = netflux_sum['olr'] + netflux_sum['circulation']
        x = 0.3
        ax1.annotate('energy in from surface: %s%s'%(round(energy_in,1),unit), xy=(x, 0.85), xycoords='axes fraction', fontsize=8, verticalalignment='top')
        ax1.annotate('energy out by OLR: %s%s (%s%%)'%(round(netflux_sum['olr'],1),unit,round(netflux_sum['olr']/energy_in*100,1)), xy=(x, 0.8), xycoords='axes fraction', fontsize=8, verticalalignment='top')
        ax1.annotate('energy out by circulation: %s%s (%s%%)'%(round(netflux_sum['circulation'],1),unit,round(netflux_sum['circulation']/energy_in*100,1)), xy=(x, 0.75), xycoords='axes fraction', fontsize=8, verticalalignment='top')
        #ax1.annotate('energy out:%s'%round(energy_out,1), xy=(1, 0.8), xycoords='axes fraction', fontsize=8, verticalalignment='top')


        #ax1.axhline(y=0.5,color='k')
        #ax1.axhline(y=1,color='k')
        ax1.plot([0,3,3,0,0],[0.5,0.5,1,1,0.5],'k')

        for i in ['left', 'right', 'top', 'bottom']:
            ax1.spines[i].set_visible(False)
            ax1.tick_params(axis='x', which='both',length=0)
            ax1.tick_params(axis='y', which='both',length=0)
            pass
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax1.set_xlim(-0.1,4.1)
        ax1.set_ylim(0,1.4)
        fname = 'energy_budget'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.2)
        plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=200, pad_inches=0.01)


