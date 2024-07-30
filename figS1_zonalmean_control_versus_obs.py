import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import ipdb
import cartopy.crs as ccrs
from importlib import reload
from copy import deepcopy

import sys; sys.path.insert(1, '/home/pyfsiew/polar_heating3')
import read_extract_data as red
import utilities as ut

if __name__ == "__main__":

    exps=['gorman_tracmip_control_rerun', 'NCEP_reanalysis_R2']

    resolution = 'climatology'; control_years, perturb_years = None, None
    season = None 

    EKE={}
    ZonalU={}
    Temp={}

    for exp in exps:
        var='ucompucomp_anom'
        upup=red.reading_vards(exp,var,years=control_years,season=season,
                        lats=None,resolution=resolution, height_lev=None, change_lons_order=False)
        var='vcompvcomp_anom'
        vpvp=red.reading_vards(exp,var,years=control_years,season=season,
                        lats=None,resolution=resolution, height_lev=None, change_lons_order=False)
        EKE[exp] = 0.5*upup+ 0.5*vpvp
        var='ucomp'
        ZonalU[exp]=red.reading_vards(exp,var,years=control_years,season=season,
                        lats=None,resolution=resolution, height_lev=None, change_lons_order=False)
        var='temp'
        Temp[exp]=red.reading_vards(exp,var,years=control_years,season=season,
                        lats=None,resolution=resolution, height_lev=None, change_lons_order=False)
        if exp=='NCEP_reanalysis_R2':
            var='surface_pressure'
            SP_NCEP=red.reading_vards(exp,var,years=control_years,season=season,
                            lats=None,resolution='monthly', height_lev=None, change_lons_order=False)
            var='temp'
            TEMP_NCEP=red.reading_vards(exp,var,years=control_years,season=season,
                            lats=None,resolution='monthly', height_lev=None, change_lons_order=False)
    SP_NCEP=SP_NCEP.compute()/100
    TEMP_NCEP=TEMP_NCEP.compute()

    if True:# Mask the subsurface data for temeprature in the reanalysis
        levs=TEMP_NCEP.lev_hPa.values
        lev_array=deepcopy(TEMP_NCEP)
        for i, lev in enumerate(levs):
            lev_array[:,i,:,:]=lev
        SP_NCEP_append=xr.concat([SP_NCEP]*len(levs),dim='lev_hPa').assign_coords(lev_hPa=levs)
        mask=SP_NCEP_append<lev_array
        TEMP_NCEP_mask=xr.where(mask,np.nan,TEMP_NCEP)
        if True: # Do the tmean in all seasons
            TEMP_NCEP_tmean=TEMP_NCEP_mask.mean(dim='time')
        else: # Do the tmean in only winter
            mon_mask=TEMP_NCEP_mask.time.dt.month.isin([1,2,3,4])
            TEMP_NCEP_tmean=TEMP_NCEP_mask.sel(time=mon_mask).mean(dim='time')

    #ipdb.set_trace()
    shading_A= [EKE[exp].mean(dim='longitude') for exp in exps]
    shading_B = [Temp[exp].mean(dim='longitude') for exp in exps]
    shading_C= [ZonalU[exp].mean(dim='longitude') for exp in exps]
    shading_grids = shading_A + shading_B + shading_C
    if True: # Replace the renalysis temperarture with masked data
        shading_grids[3]=TEMP_NCEP_tmean.mean(dim='longitude')
    rows,cols=3,2
    grid_no=rows*cols
    #mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff',
    #            '#ffffff', '#fddbc7',  '#f4a582', '#d6604d', '#b2182b']
    #cmap = matplotlib.colors.ListedColormap(mapcolors)
    cmap_grids= ['Blues']*4 + ['bwr']*2
    shading_clevels = [np.linspace(0,200,11)]*2 + [np.linspace(200,300,11)]*2 + [np.linspace(-30,30,11)]*2
    clabels_rows = ['']*grid_no
    top_titles=['']*grid_no
    ind_titles= ['Control experiment','Reanalysis'] + ['']*4
    left_titles= ['(A)', '(B)', '(C)']
    # Plot the height-latitude section
    xticklabels = [90,-60,-30,0,30,60,90]
    yticklabels = [10,200,400,600,800,1000]
    xsize=2; ysize=2
    ut.grid_plotting_height_lat(shading_grids, rows, cols, cmap_grids, shading_clevels, clabels_rows,
        top_titles=top_titles, grid=False, left_titles=left_titles, ind_titles=ind_titles, log_height=False,
        xticklabels=xticklabels, yticklabels=yticklabels, dimension='latitude',
        contour_map_grids=shading_grids, contour_clevels=shading_clevels, 
        transpose=False,xsize=xsize,ysize=ysize,title_loc='center',
        contour_label=False,xlabel='Latitude')
    fname='zonal_mean_resposnes_control_reanlaysis'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2)
    plt.savefig('/home/pyfsiew/graphs/%s_%s_ts.png' %(dt.date.today(), fname), bbox_inches='tight',
            dpi=500, pad_inches=0.01)

