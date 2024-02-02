import numpy as np
import glacier_pre_post_functions
import glob
import geopandas as gpd
import xarray as xr
import pandas as pd
import pickle as pkl
import warnings
import time

# # ------------------ POSTPROCESSING FOR WHOLE WORLD
# ------------------- ENTER YOUR PATHS ---------------
# enter path to data folder, this can be different than git folder, because data does not fit on github
#path_general = r'C:\Users\shanus\Processing\pipeline_oggm_cwatm'
path_general = "P:/watmodel/CWATM/Regions/CWatM-Otta/glaciers/"
# put in output directory
path_output_results = path_general
path_oggm_results = path_general + 'oggm_results/'


#resolution = '30min' #resolution (5min or 30min)
resolution = '1km' #resolution (1 km)
pf = 1.5   #precipitation factor
#pf = 2.0   #precipitation factor
rgi_regions = ['08']
pfname = "1"  # resuult is pf=1, because original result is dived through pf

outname = "region_{}_pf{}_{}".format(rgi_regions[0],pfname,resolution)
outnamearea = "region_area_{}_pf{}_{}".format(rgi_regions[0],pfname,resolution)

#all_regions = glob.glob(path_oggm_results + 'historical_run_output_*_1990_2019_mb_real_daily_cte_pf_{}*'.format(str(pf)))
oggm_results = []
for region in rgi_regions:
    #regionfile= path_oggm_results + '_historical_run_output_{}_1990_2019_mb_real_daily_cte_pf_{}_fixed_interpolation_temp.nc'.format(region,str(pf))
    regionfile = path_oggm_results  +"run_output_historical_daily_1979_2021_mb_real_daily_cte.nc"
    oggm_results.append(regionfile)

# -----------------------------------------------------

# example Netcf - to use the dimension of this for generating netcdfs
# cell area 30 and 5arc
#example_nc = path_general + '/cellarea/cellarea_{}.nc'.format(resolution)
# cellarea 1km
example_nc = path_general+"scripts/cellarea.nc"
#cellarea = xr.open_dataset(path_cellarea)
#path_preprocessed = path_general + '/glaciers_preprocessed/{}/'.format(resolution)


# Limit to a list of glaciers
infile = open(path_general+'scripts/id_otta_191.txt','r')
rgi_ids = infile.read()[0:-1].split(",")
rgi_ids = list(map(lambda x: "RGI60-"+rgi_regions[0]+"." + x, rgi_ids))


# --- read result preprocessing
path_output = path_output_results

# load preprocessing results - seebelow
with open(path_general+'scripts/glacier_dict2.pkl', 'rb') as handle:
    glacier_dict = pkl.load(handle)
with open(path_general+'scripts/grid_dict.pkl', 'rb') as handle:
    grid_dict = pkl.load(handle)




# ------------- GENERATE GLACIER AREA INPUT FOR CWATM -------------------
#TODO this is slow for 30arcmin global and will be very slow for 5arcmin global

start_time = time.time()

glacier_pre_post_functions.oggm_area_to_cwatm_input_world_PB(rgi_ids,glacier_dict,grid_dict, oggm_results, 1990, 2021, path_output_results,outnamearea, example_nc,resolution, fraction=True, fixed_year=None)
end_time = time.time()
print("\ntime to run whole function " + str(end_time - start_time))

# --------------- GENERATE GLACIER MELT INPUT FOR CWATM -----------------
#TODO this is very slow
start_time = time.time()
glacier_pre_post_functions.oggm_output_to_cwatm_input_world_PB(rgi_ids,glacier_dict, oggm_results, pf, 1990, 2021, path_output_results, outname, example_nc, resolution)
end_time = time.time()
print("\ntime to run whole function" + str(end_time - start_time))
#glacier_pre_post.oggm_output_to_cwatm_input_world(glacier_outlet_30min, all_regions[5:7], pf, 1990, 2019, path_output_results, " test_all_regions_mask_pf{}_{}".format(str(pf), resolution), example_nc_30arcmin, resolution, include_off_area = False)


"""
saving glacier_dict2.pkl in path_output
key: RGI-index-key e.g. RGI60-08.00503': e.g. glacier_dict2['RGI60-08.00503'][0]
[0]: array of gridcells covering this glacier
[1]: terminus point (either given or lowest point of the glacier
[2]: array of fraction of this glacier area in the gridcell(s)
[3]: array of area of this glacier in the gridcell(s)
[4]: array of elevation
[5]: array of cell-id of next downstream cell
"""
"""
saving grid_dict.pkl in path_output
key: Cell-ID  e.g. grid_dict[134695][0]
[0] RGI glaicers in this gridcell
[1] combined glacier area fraction in this gridcell
[2] total area of this gridcell
"""