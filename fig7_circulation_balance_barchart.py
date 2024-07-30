import numpy as np
import xarray as xr
import ipdb
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
from importlib import reload

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')
import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3/calculation')

import utilities as ut
import heat_budget as hb
import fig7_circulation_balance_barchart as cbb

if __name__ == "__main__":

    clim_season=''
    clim_season='winter'
    clim_season='summer'


    qheating_height, qflux_longitudes, qflux_latitudes = {}, {}, {}
    title, title_y = {}, {}
    yticks = {}
    if True: # (A):
        qflux_latitudes['A']={'75NA100L':(60,90), '60NA100L':(45,75), '45NA100L':(30,60), '30NA100L':(15,45), '15NA100L':(0,30)}
        #qflux_longitudes={'75NA100L':(-60,60),'60NA100L':(-31.1,31.1),'45NA100L':(-21.95,21.95),'30NA100L':(-17.95,17.95),'15NA100L':(-16.1,16.1)}
        qflux_longitudes['A']={'75NA100L':(-20,20),'60NA100L':(-20,20),'45NA100L':(-20,20),'30NA100L':(-20,20),'15NA100L':(-20,20)} 
        qheating_height['A']={'75NA100L':(200,1000),'60NA100L':(200,1000),'45NA100L':(200,1000),'30NA100L':(200,1000),'15NA100L':(200,1000)}
        title['A']='(A) Full troposphere'; title_y['A']=1
        yticks['A']=[-50, 0, 50,100]
    if True: # (B)
        qheating_height['B']={'75NA100L':(800,1000),'60NA100L':(800,1000),'45NA100L':(800,1000),'30NA100L':(800,1000),'15NA100L':(800,1000)}
        qflux_latitudes['B'] = {'75NA100L':(60,90), '60NA100L':(45,75), '45NA100L':(30,60), '30NA100L':(15,45), '15NA100L':(0,30)}
        qflux_longitudes['B'] = {'75NA100L':(-20,0),'60NA100L':(-20,0),'45NA100L':(-20,0),'30NA100L':(-20,0),'15NA100L':(-20,0)} 
        title['B']='(B) Low troposphere'; title_y['B']=1.05
        yticks['B']=[0,30,60,90]

    reload(cbb)
    zonal,meri,hori,vert,trans={},{},{},{},{}
    for AB in ['A','B']:
        zonal[AB],meri[AB],hori[AB],vert[AB],trans[AB]=cbb.create_balance_terms(qheating_height[AB],qflux_latitudes[AB],qflux_longitudes[AB],clim_season=clim_season)

    ### Start the plotting
    plt.close()
    fig, axs = plt.subplots(2,1,figsize=(4,4))
    x=np.arange(len(zonal['A']))
    bw=0.16
    for i, AB in enumerate(['A', 'B']):
        axs[i].bar(x-0.2,zonal[AB],width=0.075,linestyle='-',bottom=0,edgecolor='none',color='gold',zorder=5)
        axs[i].bar(x-0.12,meri[AB],width=0.075,linestyle='-',bottom=0,edgecolor='none',color='darkorange',zorder=5)
        axs[i].bar(x-0.16,hori[AB],bw,linestyle='-',bottom=0,color='red',label='Horizontal advection')
        axs[i].bar(x, vert[AB], bw, linestyle='-', bottom=0, color='royalblue', label='Vertical advection')
        axs[i].bar(x+0.16, trans[AB], bw, linestyle='-', bottom=0, color='green', label='Transient eddy')
        axs[i].set_xticks(x)
        axs[i].set_yticks(yticks[AB])
        axs[i].axhline(y=0, color='k')
        axs[i].set_title(title[AB], loc='left', size=9, x=-0.19, y=title_y[AB])
        axs[i].set_xlim(-0.5,4.5)
    axs[0].set_ylabel('Relative role (%)')
    axs[1].set_ylabel('Relative role (%)')
    axs[1].yaxis.set_label_coords(-0.13, 0.5)
    # Only for ax2
    exps=['15NA100L', '30NA100L', '45NA100L','60NA100L', '75NA100L']
    xticklabels=[exp[0:3] for exp in exps]
    axs[-1].set_xticklabels(xticklabels)
    axs[0].set_xticklabels([])
    xlabel='Heating experiment'
    axs[-1].set_xlabel(xlabel)
    # Only for ax 1
    legend_bool=True
    if legend_bool: # Add legend
        ybase=1.2
        legend1=axs[0].legend(bbox_to_anchor=(-0.2, ybase), ncol=1, loc='lower left',
                    frameon=False, columnspacing=0.5, handletextpad=0.1, labelspacing=0, prop={'size':9})
        if True: # Create second legend by some fake data
            axl = axs[0].twinx()
            axl.get_yaxis().set_visible(False)
            axl.bar(-10,-10,width=0.03,linestyle='-',bottom=0,edgecolor='gold',color='gold',label='Zonal')
            axl.bar(-10,-10,width=0.03,linestyle='-',bottom=0,edgecolor='darkorange',color='darkorange',label='Meridional')
            legend2=axl.legend(bbox_to_anchor=(0.35, ybase+0.18), ncol=2, loc='lower left',
                        frameon=False, columnspacing=0.5, handletextpad=0.1, labelspacing=0, prop={'size':9})
    #ax1.add_artist(legend1)
    for i in ['top', 'bottom', 'left', 'right']:
        for ax in axs:
            ax.spines[i].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
            axl.spines[i].set_visible(False)
            axl.tick_params(axis='x', which='both',length=2)
            axl.tick_params(axis='y', which='both',length=2)
    fname='circulation_role'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.3)
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=500, pad_inches=0.01)



def create_balance_terms(qheating_height,qflux_latitudes,qflux_longitudes, clim_season=''):

    # Read the circulation terms that contribute to the diabatic heating. Calculate their contributions in each grid point in each levels
    exps=['15NA100L', '30NA100L', '45NA100L','60NA100L', '75NA100L']
    controls=['gorman_tracmip_control_rerun']*len(exps)

    # Varying height-level
    response_terms = {}
    response_terms_weight = {}
    for exp, control in zip(exps, controls):
        exp_terms = hb.heat_budget_term(exp, clim_season=clim_season)
        control_terms = hb.heat_budget_term(control, clim_season=clim_season)
        response_terms[exp] = {t: exp_terms[t]-control_terms[t] for t in control_terms}
        response_terms_weight[exp] = {t: response_terms[exp][t]*np.cos(response_terms[exp][t].latitude*np.pi/180) 
                                for t in control_terms}
    terms = [*control_terms]

    Cp=1004 # J/K/kg
    g = 9.81
    hPa_to_Pa = 100
    response_terms_average = {}
    for exp in exps:
        # Intergrate along the height
        height1, height2 = qheating_height[exp]
        response_terms_intergral={t: response_terms_weight[exp][t].sel(lev_hPa=slice(height1,height2)).integrate('lev_hPa') * hPa_to_Pa * Cp / (24*3600) /g for t in terms}
        # Average along the box
        lat1,lat2=qflux_latitudes[exp]
        lon1,lon2=qflux_longitudes[exp]
        response_terms_average[exp]={t: response_terms_intergral[t].sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2)).mean(dim='latitude').mean(dim='longitude') for t in terms}


    zonal_advects, meri_advects, vert_advects, transients = [], [], [], []
    hori_advects = []
    for exp in exps:
        total_q=response_terms_average[exp]['u_advection']+response_terms_average[exp]['v_advection']+\
                response_terms_average[exp]['w_advection']+ response_terms_average[exp]['divergence_eddy_heatflux']
        zonal_advect= response_terms_average[exp]['u_advection']/total_q*100
        meri_advect = response_terms_average[exp]['v_advection']/total_q*100
        hori_advect = zonal_advect+meri_advect
        vert_advect = response_terms_average[exp]['w_advection']/total_q*100
        transient = response_terms_average[exp]['divergence_eddy_heatflux']/total_q*100
        zonal_advects.append(zonal_advect)
        meri_advects.append(meri_advect)
        hori_advects.append(hori_advect)
        vert_advects.append(vert_advect)
        transients.append(transient)
        print(exp, zonal_advect+meri_advect+vert_advect+transient)

    return zonal_advects, meri_advects, hori_advects, vert_advects, transients


