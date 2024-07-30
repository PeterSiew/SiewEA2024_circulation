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
from scipy.stats import pearsonr

import read_extract_data as red

if __name__ == "__main__":

    reload(hb); reload(red)

    exps=['15NA100L','30NA100L','45NA100L','60NA100L','75NA100L']
    control='gorman_tracmip_control_rerun'
    clim_season='winter'
    clim_season='summer'
    clim_season=''

    plotting_type='lat_lon'
    plotting_type='height_lon'
    dim_value=500
    dim_value=slice(37.5,52.5) # for 45N heating
    dim_value={'15NA100L':slice(0,30),'30NA100L':slice(15,45),'45NA100L':slice(30,60),'60NA100L':slice(45,75),
            '75NA100L':slice(60,90)}

    ### Select what to calculate
    control_terms = hb.heat_budget_term(control, clim_season=clim_season)
    response_term_sel = {exp:{} for exp in exps}
    for exp in exps:
        exp_terms = hb.heat_budget_term(exp, clim_season=clim_season)
        for term in control_terms.keys():
            response_term = exp_terms[term] - control_terms[term]
            response_term = red.select_dim(response_term, plotting_type='height_lon', dim_value=dim_value[exp])
            # Change the longtiude dimension to -180 to 180
            longitude = response_term.longitude.values
            new_longitude = [lon if lon<=180 else lon-360 for lon in longitude]
            response_term = response_term.assign_coords(longitude=new_longitude).sortby('longitude')
            response_term_sel[exp][term] = response_term

    shading_grids = []
    #corrs = []
    #lons ={'15NA100L':16,'30NA100L':18,'45NA100L':22,'60NA100L':31,'75NA100L':60}
    #heights ={'15NA100L':200,'30NA100L':300,'45NA100L':400,'60NA100L':500,'75NA100L':600}
    terms = ['Q', 'w_advection', 'u_advection', 'v_advection', 'divergence_eddy_heatflux'] # Manual set the orders. Otherwise control_terms.keys()
    ratio = {'Q':1, 'u_advection':-1, 'v_advection':-1, 'w_advection':-1, 'divergence_eddy_heatflux':-1}
    for term in terms:
        for exp in exps:
            shading_grids.append(response_term_sel[exp][term]*ratio[term])
            if False: # For correlations
                Q_term = response_term_sel[exp]['Q'].sel(longitude=slice(lons[exp]*-1,lons[exp])). \
                        sel(lev_hPa=slice(heights[exp],1000)).values.reshape(-1)
                test_term= response_term_sel[exp][term].sel(longitude=slice(lons[exp]*-1,lons[exp])). \
                        sel(lev_hPa=slice(heights[exp],1000)).values.reshape(-1)
                corr=pearsonr(Q_term, test_term)
                print(term, exp, corr[0])
                corrs.append(round(corr[0],2))
                freetext = corrs

    shading_clevels = [np.linspace(-1,1,11)] * len(shading_grids)
    contour_grids = shading_grids
    contour_clevels = shading_clevels

    # Make the plotting
    cols = len(exps)
    rows = len(control_terms.keys())
    grids = rows*cols
    clabels_rows = [''] * rows
    ABC = ['A', 'B', 'C', 'D', 'E', 'F']
    ind_titles = ['('+ABC[i]+') '+exp[0:3] for i, exp in enumerate(exps)] + [None]*(rows-1)*cols
    top_titles = [None]*cols
    freetext = None
    freetext_pos = [(0.001,0.001)] * grids
    pval_map = None
    pval_hatches = None
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff', '#ffffff', '#fddbc7',  '#f4a582', '#d6604d', '#b2182b']
    cmap = matplotlib.colors.ListedColormap(mapcolors)
    mapcolors_grids = [cmap] * grids
    xsize=3; ysize=1.5
    xticklabels = [-180,-120,-60, 0, 60, 120, 180]
    xticklabels = [-60,-30, 0, 30, 60]
    yticklabels = [10,200,400,600,800,950]
    yticklabels = [200,400,600,800,950]
    plot_dim='longitude'

    terms_equation={'u_advection':r"$u\frac{\partial \theta}{\partial x}$",
					'v_advection':r"$v\frac{\partial \theta}{\partial y}$",
					'w_advection':r"$w\frac{\partial \theta}{\partial p}$",
					'ddx_utheta':r"$ \frac{\partial}{\partial x} (u'\theta')$",
					'ddy_vtheta':r"$ \frac{\partial}{\partial y} (v'\theta')$",
					'ddz_wtheta':r"$ \frac{\partial}{\partial z} (w'\theta')$",
					'divergence_eddy_heatflux':r"$ \frac{\partial}{\partial x}(u'\theta')$"+ \
                    '\n+\n' r"$ \frac{\partial}{\partial y} (v'\theta')$"+ \
					 r"$ \frac{\partial}{\partial z} (w'\theta')$",
					'Q':r"$Q$"}
    #left_titles = [terms['u_advection'],terms['v_advection'],terms['w_advection'],terms['divergence_eddy_heatflux'],terms['Q']]
    left_titles_keys={'u_advection':'Zonal\ntemperature\nadvection','v_advection':'Meridional\ntemperature\nadvection',
            'w_advection':'Vertical\npotential\ntemperature\nadvection',
            'divergence_eddy_heatflux':'Transient\neddy\nheat flux\ndivergence','Q':'Diabatic\nheating'}
    left_titles = [left_titles_keys[term] for term in terms]

    fig=ut.grid_plotting_height_lat(shading_grids, rows, cols, mapcolors_grids, shading_clevels, clabels_rows,
        top_titles=top_titles, left_titles=left_titles,pval_map=pval_map, pval_hatches=pval_hatches,
        contour_map_grids=contour_grids, contour_clevels=contour_clevels, ind_titles=ind_titles,
        log_height=False, xticklabels=xticklabels, yticklabels=yticklabels, xsize=xsize, ysize=ysize,dimension=plot_dim,
        freetext=freetext, freetext_pos=freetext_pos, transpose=False, contour_label=False,colorbar=False)

    if True: ### Setup the colorbar
        cba = fig.add_axes([0.2, 0.01, 0.6, 0.03]) 
        cNorm  = matplotlib.colors.Normalize(vmin=shading_clevels[0][0], vmax=shading_clevels[0][-1])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
        xticks = xticklabels = [round(i,2) for i in shading_clevels[0] if i!=0]
        cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='horizontal',ticks=xticks, extend='both')
        cb1.ax.set_xticklabels(xticklabels, fontsize=10)
        cb1.set_label('', fontsize=10, x=1.12, labelpad=-22)

    fname = 'heatbudget_new'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2)
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight', dpi=500)
