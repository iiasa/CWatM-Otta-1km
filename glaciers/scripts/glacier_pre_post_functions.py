import pandas as pd
import pickle
import xarray as xr
import rasterio
import numpy as np
import netCDF4 as nc
import os
import math
import warnings
import glob
import geopandas as gpd
import time
#from memory_profiler import profile
import datetime as dt

import add_functions
# -------------------------- FUNCTIONS ------------------------------------


#compare glaciers that are in area with glaciers_geodetic und benutze die schnittmenge

#function that gives the areal extent of glaciers at the rgi date in each grid cell
'''maybe this can be preprocessed for the whoel world and then the function just gets the values of the current glaciers'''


def transform_to_df_PB(dict_glacier, overlay_result, resolution):

    df = pd.DataFrame()
    #rgiid
    df["RGIId"] = overlay_result.RGIId
    if resolution == "1km":
        df["Nr_Gridcell"] = overlay_result.Id.astype(int)
        df["Latitude"] = overlay_result.y
        df["Longitude"] = overlay_result.x
        df["area"] = overlay_result.Area_Cell
        df["area_fraction"] = overlay_result.area_frac
        demfile = "P:/watmodel/OGGM/pipeline_oggm_cwatm/glaciers_preprocessed_peter/1km/dem_min.tif"
        lddfile = "P:/watmodel/OGGM/pipeline_oggm_cwatm/glaciers_preprocessed_peter/1km/ldd_add.tif"

    else:
        df["Nr_Gridcell"] = overlay_result.FID.astype(int)
        if resolution == "5min":
            df["Latitude"] = np.round(overlay_result.lat, decimals=5)
            df["Longitude"] = np.round(overlay_result.lon, decimals=5)
            demfile = "P:/watmodel/OGGM/pipeline_oggm_cwatm/glaciers_preprocessed_peter/5min/dem_min.tif"
            lddfile = "P:/watmodel/OGGM/pipeline_oggm_cwatm/glaciers_preprocessed_peter/5min/ldd.nc"

        else:
            df["Latitude"] = overlay_result.lat
            df["Longitude"] = overlay_result.lon
            demfile = "P:/watmodel/OGGM/pipeline_oggm_cwatm/glaciers_preprocessed_peter/30min/dem_min.tif"
            lddfile = "P:/watmodel/OGGM/pipeline_oggm_cwatm/glaciers_preprocessed_peter/30min/ldd.nc"

        df["area"] = overlay_result.Area_Cell
        df["area_fraction"] = overlay_result.area_frac

    # cellno is index to choose right grid cell fitting to RGI-overlay cell
    cellno = df["Nr_Gridcell"].to_numpy(dtype=int)

    dem = rasterio.open(demfile).read()[0, :, :]
    demflat = dem.flatten(order="F")
    df["dem"] = demflat[cellno]

    ldd = rasterio.open(lddfile).read()[0, :, :]
    lddflat = ldd.flatten(order="F")
    lddcell = lddflat[cellno]

    # calculate next cell from ldd
    if resolution == "1km":
        transldd = np.array([0, -199, 1, 201, -200, 0, 200, -201, -1, 199], dtype=int)
    else:
        # only global 30min
        transldd = np.array([0, -359, 1, 361, -360, 0, 360, -361, -1, 359], dtype=int)
    moveldd = transldd[lddcell]
    df["ldd"] = cellno + moveldd

    # sort df by elevation - high to low -> for putting or removing glacier area depending on elevation
    df = df.sort_values(by=['dem'], ascending=False)
    #df = df.sort_values(by=['RGIId'])
    df.set_index("RGIId", inplace=True)

    # Glacier dictionary for each RGI glacier has array of gridcells
    glaciers_dict2 = dict()

    for ind in df.index:
        if ind in dict_glacier.keys():
            gridcells = df.loc[ind, "Nr_Gridcell"]
            terminus = dict_glacier[ind][0]
            #print (gridcells," ->",terminus)

            if (type( gridcells) is pd.core.series.Series):
                # if glacier has cell in different grids
                cells = gridcells.tolist()
                if (terminus in cells):
                    index = cells.index(terminus)
                else:
                    # use lowest DEM if not terminus is found
                    index = -1
                term = int(df.loc[ind, "Nr_Gridcell"][index])
                cell = np.asarray(df.loc[ind, "Nr_Gridcell"],dtype=int)
                area_fraction = np.asarray(df.loc[ind, "area_fraction"])
                area = np.asarray(df.loc[ind, "area"])
                dem = np.asarray(df.loc[ind, "dem"])
                ldd = np.asarray(df.loc[ind, "ldd"],dtype=int)

            else:
                # only 1 cell in grid
                term = int(df.loc[ind, "Nr_Gridcell"])
                cell = np.asarray([df.loc[ind, "Nr_Gridcell"]], dtype=int)
                area_fraction = np.asarray([df.loc[ind, "area_fraction"]])
                area = np.asarray([df.loc[ind, "area"]])
                dem = np.asarray([df.loc[ind, "dem"]])
                ldd = np.asarray([df.loc[ind, "ldd"]], dtype=int)

            df.loc[ind, "cellnr_terminus"] = term
            if ind not in glaciers_dict2:
                glaciers_dict2.update({ind: [cell, term, area_fraction, area, dem, ldd]})
                # later on if glacier is growing - glaciers_dict2['RGI60-08.00338'][0] = np.append(glaciers_dict2['RGI60-08.00338'][0],2222)
            else:
                iii = 0
                # should not go here
    return glaciers_dict2

def create_griddict_PB(glacier_dict2, areagrid, resolution):

    # grid dictornary has array of RGI glaciers and combined area_fraction
    grid_dict = dict()

    for glacier in glacier_dict2:

        cell = glacier_dict2[glacier][0]
        for i, c in enumerate(cell):
            if c not in grid_dict:
                area = areagrid[c]
                grid_dict.update({c: [[glacier], glacier_dict2[glacier][2][i],area ]})
            else:
                # add RGI glacier and sum up total glacier area in this cell
                grid_dict[c][0].append(glacier)
                grid_dict[c][1] += glacier_dict2[glacier][2][i]

    return grid_dict



def df_glacier_grid_area_PB(glaciers_dict, grid, rgi_regions, path_glacier_info, name_glacier_info, path_rgi_files, name_output, resolution):

    if resolution == "1km":
        filename = path_rgi_files + '/rgi_region_{}_{}.shp'.format(rgi_regions[0],resolution)
        if os.path.isfile(filename):
            overlay = gpd.read_file(filename)
            areagrid = grid.area
        else:
            rgi_file = path_rgi_files + "rgi_otta191_utm33" + '.shp'
            rgi = gpd.read_file(rgi_file)
            # results from overlay functions
            overlay_result = gpd.overlay(rgi, grid, how='intersection')
            # calculate area fraction of the overlayed glacier of total grid cell
            area = overlay_result.area
            # calculate area fraction of the overlayed glacier of total grid cell
            area = overlay_result.area
            overlay_result['area_frac'] = area / 1000000.
            # #get area of each part
            overlay_result["Area_Cell"] = overlay_result.area

            overlay_result.to_file(path_glacier_info + '/rgi_region_{}_{}.shp'.format(rgi_regions[0], resolution))
            # get the area of gridcells
            areagrid = grid.area



            glacier_dict2 = transform_to_df_PB(glaciers_dict, overlay_result, resolution)
            grid_dict = create_griddict_PB(glacier_dict2, areagrid, resolution)

            with open(path_glacier_info + 'glacier_dict2.pkl', 'wb') as handle:
                pickle.dump(glacier_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path_glacier_info + 'grid_dict.pkl', 'wb') as handle:
                pickle.dump(grid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # for 5 and 30 arcmin
    else:
        for i, current_rgi in enumerate(rgi_regions):
            #print(current_rgi)
            overlay_54012_filename = path_glacier_info + '/rgi_region_{}_{}.shp'.format(current_rgi,resolution)
            if os.path.isfile(overlay_54012_filename):
                overlay_54012 = gpd.read_file(overlay_54012_filename)
                grid_54012 = grid.to_crs('esri:54012')
                areagrid = grid_54012.area
            else:
                rgi_file = glob.glob(path_rgi_files + current_rgi + '*/*.shp')[0]
                rgi = gpd.read_file(rgi_file)

                # results from overlay functions
                overlay_result = gpd.overlay(rgi, grid, how='intersection')
                # calculate area fraction of the overlayed glacier of total grid cell
                area = overlay_result.area
                if resolution =="30min":
                    overlay_result['area_frac']= area / 0.25
                else: # 5min
                    overlay_result['area_frac'] = area / 0.006944

                # #get area of each part
                overlay_54012 = overlay_result.to_crs('esri:54012')
                overlay_54012["Area_Cell"] = overlay_54012.area
                overlay_54012.to_file(path_glacier_info + '/rgi_region_{}_{}.shp'.format(current_rgi, resolution))

                # get the area of gridcells
                grid_54012 = grid.to_crs('esri:54012')
                areagrid = grid_54012.area

            if i == 0:
                glacier_dict2 = transform_to_df_PB(glaciers_dict, overlay_54012, resolution)
                grid_dict = create_griddict_PB(glacier_dict2, areagrid, resolution)
            else:
                glacier_dict2 = pd.concat([glacier_dict2, transform_to_df_PB(glaciers_dict, overlay_54012, resolution)])
                grid_dict = pd.concat([grid_dict, create_griddict_PB(glacier_dict2,areagrid, resolution)])
            ii =1
            with open(path_glacier_info + 'glacier_dict2.pkl', 'wb') as handle:
                pickle.dump(glacier_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path_glacier_info + 'grid_dict.pkl', 'wb') as handle:
                pickle.dump(grid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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



    #with open('filename.pickle', 'rb') as handle:
    #    unserialized_data = pickle.load(handle)

    #df_all.to_csv(path_glacier_info+name_output+'_{}.csv'.format(resolution))
    #df_all.to_pickle(path_glacier_info+name_output+"glacier_info_{}.pkl".format(resolution))


def make_glacier_outlet_dict_PB(list_path_glacierstats,path_output, resolution, rgi_ids=None):
    '''generates the glacier outlet dictionary with the glacier_outlet_dict function
    list_path_glacierstats: list of paths with rgi glacier statistics
    outpath:
    out_name:
    resolution: '5min' or '30min' or 1km
    rgi_ids: if rgi ids are given only generates it the dictionary for the given rgi_ids
    '''

    glaciers_dict = dict()
    # read from glacier statistics -> terminus of each glacier
    for i, path_rgi_reg in enumerate(list_path_glacierstats):
        glaciers = pd.read_csv(path_rgi_reg, low_memory=False)

        # use only glacier defined in list or use all of the region
        if rgi_ids:
            glacier_stats_basin = glaciers[np.isin(glaciers.rgi_id, rgi_ids)]
        else:
            glacier_stats_basin = glaciers

    for index, row in glacier_stats_basin.iterrows():
        if resolution == "1km":
            y = row.y
            x = row.x
            # hardcoded and based on Norway Otta grid
            terminus_cell =int((x-50000) / 1000) * 200 + int((6950000-y)/1000)
            glaciers_dict.update({row.rgi_id: [terminus_cell,(y,x),row.rgi_area_km2,row.dem_min_elev]})
        else:
            lat = row.terminus_lat
            lon = row.terminus_lon
            # hardcoded and based on Sarah's grid
            lat_int = int((90-lat) * 2)
            lon_int = int((180+lon) * 2)
            terminus_cell =  lon_int * 360 + lat_int
            glaciers_dict.update({row.rgi_id: [terminus_cell,(lat,lon),row.rgi_area_km2,row.dem_min_elev]})

    # store terminus cell as cell-id, lat,lon, glacier area and min elevation
    with open(path_output+'glacier_dict.pkl', 'wb') as handle:
        pickle.dump(glaciers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return glaciers_dict














#-------- POSTPROCESSING --------------------

def change_format_oggm_output(glacier_run_results_list, variable):
    '''Function changes the format of oggm output from 2d array (days,years) to a 1d timeseries'''
    #get start date and end date of OGGM runs
    for idx, glacier_run_results in enumerate(glacier_run_results_list):
        startyear = glacier_run_results.calendar_year.values[0]
        startmonth = glacier_run_results.calendar_month.values[0]
        endyear = glacier_run_results.calendar_year.values[-1]
        endmonth = glacier_run_results.calendar_month.values[-1]
        #construct a string of start time and endtime
        starttime = str(startyear) + '-' + str(startmonth) + '-01'
        endtime = str(endyear) + '-' + str(endmonth) + '-01'
        #construct a timeseries with daily timesteps
        #last day is not used because the data of 2020 is not there
        timeseries = pd.date_range(starttime, endtime, freq="D")[:-1]
        #concatenate the arrays
        #concatenate current year with next year
        # the last year in OGGM results is not used because it does not give proper results
        #TODO understand why this is
        for i in range(0,len(glacier_run_results.calendar_year)-2):
            if i == 0:
                all_months = np.concatenate((glacier_run_results[variable][i].values, glacier_run_results[variable][i+1].values),axis=0)
            else:
                all_months = np.concatenate((all_months, glacier_run_results[variable][i+1].values), axis=0)

        if idx == 0:
            all_months_all = all_months
        else:
            # TODO all months next has to be concatenated
            all_months_all = np.concatenate((all_months_all,all_months),axis=1)

            #delete the nan values on the 29th of february
    all_months_all = all_months_all[~np.isnan(all_months_all).any(axis=1), :]
    #assert that the timeseries and new dara have same length
    assert len(timeseries) == np.shape(all_months_all)[0]
    return all_months_all, timeseries




def change_format_oggm_output_world_PB(rgi_ids,oggm_results_list, startyear_df, endyear_df, variable):

    for idx, oggm_results_path in enumerate(oggm_results_list):
        print(oggm_results_path)

        oggm_results = xr.open_dataset(oggm_results_path)
        nok = oggm_results.volume.isel(time=0).isnull()
        # drop rgi_ids with nan values, so RGI IDs that were not correctly modelled in OGGM
        # the threshold was set to 10 days because end of modelling period it is always 0
        # check if OGGM results contains glaciers with missing results
        if np.count_nonzero(nok) > 0: #len(oggm_results.rgi_id) != len(oggm_results_new.rgi_id):
            oggm_results_new = oggm_results.dropna('rgi_id', thresh=10)
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results.rgi_id.values)) - set(list(oggm_results_new.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results.rgi_id) - len(oggm_results_new.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
            oggm_results.close()
            del oggm_results
            oggm_results = oggm_results_new
            del oggm_results_new

        # select only those one in a list
        if not(rgi_ids == None):
            oggm_results = oggm_results.sel(rgi_id=rgi_ids)



        startyear = oggm_results.calendar_year.values[0]
        startmonth = oggm_results.calendar_month.values[0]
        endyear = oggm_results.calendar_year.values[-1]
        endmonth = oggm_results.calendar_month.values[-1]
        #construct a string of start time and endtime
        starttime = str(startyear) + '-' + str(startmonth) + '-01'
        endtime = str(endyear) + '-' + str(endmonth) + '-01'
        #construct a timeseries with daily timesteps
        #last day is not used because the data of 2020 is not there
        timeseries = pd.date_range(starttime, endtime, freq="D")[:-1]
        #concatenate the arrays
        #concatenate current year with next year
        # the last year in OGGM results is not used because it does not give proper results
        diff_start = startyear_df - startyear
        diff_end = endyear - endyear_df
        starttime = str(startyear_df) + '-01-01'
        endtime = str(endyear_df) + '-12-31'
        timeseries = pd.date_range(starttime, endtime, freq="D")
        melt_below_zero = 0

        #TODO understand why this is
        for i in range(diff_start, diff_start + endyear_df - startyear_df + 1):
        #for i in range(0,len(oggm_results.calendar_year)-1):
            melt_year = oggm_results['melt{}'.format(variable)][i].values
            if np.min(melt_year) < 0:
                melt_below_zero = 1
                msg = 'Original OGGM results has some days with melt below 0, this is due to mass conservation scheme in OGGM. melt was clipped to minimum 0'
                warnings.warn(msg)
            # CLLIP TO ZERO TO AVOID SMALL NEGATIVE VALUES
            #these negative values are due to mass conservation in oggm
            melt_year = melt_year.clip(min=0)
            rain_year = oggm_results['liq_prcp{}'.format(variable)][i].values
            if i == diff_start:
                all_months_melt = melt_year.copy() #np.concatenate((melt_year, melt_year),axis=0)
                all_months_rain = rain_year.copy() #np.concatenate((rain_year, rain_year), axis=0)
            else:
                all_months_melt = np.concatenate((all_months_melt, melt_year), axis=0)
                all_months_rain = np.concatenate((all_months_rain, rain_year), axis=0)


        if idx == 0: # loop through different RGI regions
            glacier_ids = list(oggm_results.rgi_id.values)
        else:
            glacier_ids += list(oggm_results.rgi_id.values)
        oggm_results.close()
        del oggm_results

        if idx == 0:
            all_months_melt_all = all_months_melt.copy()
            all_months_rain_all = all_months_rain.copy()
        else:
            # TODO all months next has to be concatenated
            all_months_melt_all = np.concatenate((all_months_melt_all,all_months_melt),axis=1)
            all_months_rain_all = np.concatenate((all_months_rain_all, all_months_rain), axis=1)

            #delete the nan values on the 29th of february
    all_months_melt_all = all_months_melt_all[~np.isnan(all_months_melt_all).any(axis=1), :]
    all_months_rain_all = all_months_rain_all[~np.isnan(all_months_rain_all).any(axis=1), :]
    #assert that the timeseries and new dara have same length
    assert len(timeseries) == np.shape(all_months_melt_all)[0]
    assert len(timeseries) == np.shape(all_months_rain_all)[0]
    return all_months_melt_all, all_months_rain_all, glacier_ids, timeseries




def oggm_output_to_cwatm_input(glacier_outlet, oggm_results_org, pf, startyear, endyear, outpath, out_name, example_netcdf, resolution, include_off_area = False, melt_or_prcp = 'melt'):
    '''
    Function generates a netcdf of daily glacier melt using OGGM outputs. The outlet point of each glacier is fixed to the gridcell of the terminus as in RGI
        and will not change even if glacier retreats, as the effect is assumed to be minimal.

        glacier_outlet: a dictionary of all glaciers worldwide with keys coordinates of gridcells and values a list of glacier ids which drain into the gridcell
        (CAREFUL: you might want to adapt it to fit to the basin)
        oggm_results: netcdf file of daily results of an OGGM run
        pf: precipitation factor which was used to generate OGGM results
        startyear: startyear for which to create netcdf
        endyear: endyear for which to create netcdf
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :example_netcdf: example netcdf with same extent that you want to generate

        returns: metcdf with daily glacier melt (m3/d)
    '''
    if melt_or_prcp not in ["melt", "prcp"]:
        raise ValueError("melt_or_prcp should be melt or prcp")
    if melt_or_prcp == 'melt':
        var_name = 'melt'
        #pf = 1
    elif melt_or_prcp == 'prcp':
        var_name = 'liq_prcp'
        #pf = pf

    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    # drop rgi_ids with nan values, so RGI IDs that were not correctly modelled in OGGM
    # the threshold was set to 10 days because end of modelling period it is always 0
    #TODO this has to work for list of oggm resuults
    if isinstance(oggm_results_org, list):
        oggm_results = []
        for i, oggm_results_org_part in enumerate(oggm_results_org):
            # drop rgi_ids with nan values
            oggm_results_part = oggm_results_org_part.dropna('rgi_id', thresh=10)
            # check if OGGM results contains glaciers with missing results
            if len(oggm_results_org_part.rgi_id) != len(oggm_results_part.rgi_id):
                # get RGI IDs of missing values
                rgi_ids_nan = list(
                    set(list(oggm_results_org_part.rgi_id.values)) - set(list(oggm_results_part.rgi_id.values)))
                # get number of glaciers with nan values
                missing_rgis = len(oggm_results_org_part.rgi_id) - len(oggm_results_part.rgi_id)
                # print warning that netcdf will not be generated for the glaciers with missing OGGM results
                msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                    missing_rgis, rgi_ids_nan)
                warnings.warn(msg)
            oggm_results.append(oggm_results_part)
            if i == 0:
                glacier_ids = list(oggm_results_part.rgi_id.values)
            else:
                glacier_ids += list(oggm_results_part.rgi_id.values)
    else:
        # drop rgi_ids with nan values
        oggm_results = oggm_results_org.dropna('rgi_id', thresh=10)
        # check if OGGM results contains glaciers with missing results
        if len(oggm_results_org.rgi_id) != len(oggm_results.rgi_id):
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results_org.rgi_id.values)) - set(list(oggm_results.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results_org.rgi_id) - len(oggm_results.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
        #get ids of glaciers of oggm run output
        glacier_ids = list(oggm_results.rgi_id.values)
        oggm_results = [oggm_results]

    # define extent for which glacier maps should be created
    if type(example_netcdf) == str:
        example_nc = xr.open_dataset(example_netcdf)
        lat = np.round(example_nc.lat.values,
                       decimals=5)  # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = np.round(example_nc.lon.values,
                       decimals=5) # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth = np.round((lat[0] - lat[-1]) / (len(lat)-1), decimals = 5) #lat[0] -lat[1]
        cellnr_lat = len(lat)
        cellnr_lon = len(lon)
    #if you do not have an example netcdf file you need the cellsize, the number of cells and the coordinates in upper left corner
    elif len(example_netcdf) == 5:
        lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth = example_netcdf
        lon_max = lon_min + (cellnr_lon * cellwidth)
        lat_min = lat_max - (cellnr_lat * cellwidth)
        lon = np.round(np.arange(lon_min + cellwidth/2, lon_max, cellwidth), decimals=5)
        # #start latitude from -60 because there is only ocean below anyways
        lat = np.round(np.arange(lat_min + cellwidth/2, lat_max, cellwidth), decimals=5)
        cellnr_lon = np.round(cellnr_lon, decimals=5)
        cellnr_lat = np.round(cellnr_lat, decimals=5)
        cellwidth = np.round(cellwidth, decimals=5)

    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1/12
    np.testing.assert_almost_equal(cellwidth, cellwidth_res, decimal=4,
                                   err_msg='example_nc and resolution need to have same resolution', verbose=True)


    #change output of OGGM from 2darray (years, daysofyear, glaciers) to continous timeseries
    flux_on_glacier, timeseries_glacier = change_format_oggm_output(oggm_results, '{}_on_glacier_daily'.format(var_name))
    if include_off_area:
        flux_off_glacier, _ = change_format_oggm_output(oggm_results, '{}_off_glacier_daily'.format(var_name))

    #get start and end date corresponding to the inputs of the function
    start_index = np.where(timeseries_glacier.year == startyear)[0][0]
    end_index = np.where(timeseries_glacier.year == endyear)[0][-1]
    timeseries = pd.date_range(timeseries_glacier[start_index].strftime('%Y-%m'), timeseries_glacier[end_index],freq='D')

    #create netcdf file for each variable (maybe only do this for melt_on, liq_prcp_on because we do not need the others)
    name = '{}_on'.format(var_name)
    if include_off_area:
        ds = nc.Dataset(outpath +name[:-2]  + 'total_'+ out_name +'.nc', 'w', format='NETCDF4')
    else:
        ds = nc.Dataset(outpath + name+ '_' + out_name + '.nc', 'w', format='NETCDF4')
    # add dimenstions, specify how long they are
    # use 0.5° grid
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', None)
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"
    #create variables glacier melt on off, liquid precipitation on off
    if include_off_area:
        #var_nc = ds.createVariable(name[:-2] + 'total', 'f4', ('time', 'lat', 'lon',),zlib=True, least_significant_digit=1)
        # PB with chunk
        var_nc = ds.createVariable(name[:-2] + 'total', 'f4', ('time', 'lat', 'lon',), chunksizes=(1,len(lat),len(lon)), fill_value=1e20,zlib=True,
                                   least_significant_digit=1)
    else:
        #var_nc = ds.createVariable(name, 'f4', ('time', 'lat', 'lon',),zlib=True, least_significant_digit=1)
        # PB
        var_nc = ds.createVariable(name, 'f4', ('time', 'lat', 'lon',), chunksizes=(1,len(lat),len(lon)), fill_value=1e20, zlib=True, least_significant_digit=1)

    var_nc.units = "m3/d"
    #use the extent of the example netcdf
    #TODO maybe no example netcdf needed but it can be done from scratch
    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)
    timeseries_str = str(timeseries)
    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    #now total melt input for timeseries
    #output is in kg/day -> transform to m3/day by dividing by 1000
    glacier_flux_on = flux_on_glacier[start_index:end_index+1]
    if include_off_area:
        glacier_flux_off = flux_off_glacier[start_index:end_index + 1]
        glacier_flux_on = glacier_flux_on / 1000 / pf + glacier_flux_off / 1000 / pf
    else:
        glacier_flux_on = glacier_flux_on / 1000 / pf

    # get gridcells of glaciers in right format
    gridcell_glaciers = list(glacier_outlet.keys())
    #get latitude and longitude of grid cells into which glaciers drain
    grid_lat = [a_tuple[0] for a_tuple in gridcell_glaciers]
    grid_lon = [a_tuple[1] for a_tuple in gridcell_glaciers]

    #define dataframe to store results, index are the days in timeseries
    df_flux_on = pd.DataFrame(index=timeseries)

    # TODO: see if it makes sense to mask outlet by correct basin outline
    x = np.zeros((cellnr_lat, cellnr_lon))
    x = x.astype(int)
    glacier_flux_on_array = x
    glacier_flux_on_array = glacier_flux_on_array.flatten()
    assert len(timeseries) == np.shape(df_flux_on)[0]


    #loop through all gridcells with glaciers in basin and sum up the melt of all glaciers in each grid cell
    #if glacier_outlet for the whole world, first constrain it to basin
    keys_gridcells = list(glacier_outlet)
    list_rgi_ids_world = list(glacier_outlet.values())
    count_gl_not_oggm_results = 0
    for i, rgi_ids_world in enumerate(list_rgi_ids_world):
        if np.isin(rgi_ids_world, glacier_ids).any():
            grid_lat = keys_gridcells[i][0]
            grid_lon = keys_gridcells[i][1]
            ids_gridcell = glacier_outlet[keys_gridcells[i]]


    # for gridcell in range(len(grid_lat)):
            daily_flux_on = 0
        #loop through all glaciers in the gridcell by getting the items in the dict of this grid cell
            for id in ids_gridcell: #list(glacier_outlet.items())[gridcell][1]:
                #assert that there are no nan values
                #assert np.sum(np.nonzero(np.isnan(glacier_melt[:, glacier_ids.index(id)]))) == 0
                #if there are no nan values in timeseries
                # if id o

                if id not in glacier_ids:
                    #if glacier id of glacier that drains into the basin is not modelled in OGGM, raise ERROR
                    # THIS CAN BE TURNED OFF, IF YOU ONLY WANT TO MODEL SOME GLACIERS OR SO
                    #TODO make this better
                    msg = 'Glacier {} was not found in OGGM results'.format(id)
                    #raise ValueError(msg)
                    warnings.warn(msg)
                    count_gl_not_oggm_results += 1
                    #if no nan values exist, then sum up the variables of all glaciers in the gridcell
                elif np.sum(np.nonzero(np.isnan(glacier_flux_on[:, glacier_ids.index(id)]))) == 0:
                    # sum up timeseries of all glaciers in gridcell
                    daily_flux_on += glacier_flux_on[:, glacier_ids.index(id)]
                else:
                    raise ValueError('Nan values encountered in timeseries of {} for variable {}'.format(id, glacier_flux_on))

            #daily melt volumes have to be stored in a datafram with column names being the gridcell lat, lon
            round_lat = np.round(grid_lat, decimals=3)
            round_lon = np.round(grid_lon, decimals=3)

            cell_lat = (np.max(lat) - round_lat) / cellwidth
            cell_lon = (round_lon - np.min(lon)) / cellwidth
            if cell_lat % 1 <  0.1 or cell_lat % 1 >  0.9:
                cell_lat = int(np.round(cell_lat))
            else:
                print(cell_lat)
                raise ValueError
            if cell_lon % 1 < 0.1  or cell_lon % 1 >  0.9:
                cell_lon = int(np.round(cell_lon))
            else:
                print(cell_lon)
                raise ValueError

            ind_cell = (cell_lat) * cellnr_lon + cell_lon
            if ind_cell <= len(glacier_flux_on_array):
                df_flux_on[ind_cell] = daily_flux_on
            else:
                warnings.warn("The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")
    glacier_gridcell_index = df_flux_on.columns
    if count_gl_not_oggm_results != 0:
        warnings.warn(
            "{} glaciers were not found in OGGM results but are in a grid cell for which other glaciers were modelled in OGGM. Check carefully".format(count_gl_not_oggm_results))

    for i in range(len(timeseries)):
        glacier_flux_on_array[glacier_gridcell_index] = df_flux_on.iloc[i, :].values
        # glacier_flux_on[glacier_ids] = np.ones(len(glacier_ids))
        glacier_flux_on_2d = np.reshape(glacier_flux_on_array, (cellnr_lat, cellnr_lon))
        var_nc[i, :, :] = glacier_flux_on_2d
    ds.close()









def oggm_output_to_cwatm_input_world_PB(rgi_ids,glacier_dict, oggm_results_path, pf_sim, startyear, endyear, outpath, out_name, example_netcdf, resolution):

    lat, lon, cellwidth,cellnr_lat,cellnr_lon, cellwidth_res = add_functions.read_example_nc(example_netcdf, resolution)
    #cellwidth,cellnr_lat,cellnr_lon, cellwidth_res -> nc_info

    #change output of OGGM from 2darray (years, daysofyear, glaciers) to continous timeseries
    melt_on_glacier, rain_on_glacier, glacier_ids, timeseries_glacier = change_format_oggm_output_world_PB(rgi_ids,oggm_results_path, startyear, endyear, '_on_glacier_daily')

    #get start and end date corresponding to the inputs of the function
    start_index = np.where(timeseries_glacier.year == startyear)[0][0]
    end_index = np.where(timeseries_glacier.year == endyear)[0][-1]
    timeseries = pd.date_range(timeseries_glacier[start_index].strftime('%Y-%m'), timeseries_glacier[end_index],freq='D')
    vars = ['melt', 'liq_prcp']
    #pfs = [1, pf_sim]
    pfs = [pf_sim, pf_sim]

    for k, flux_on_glacier in enumerate([melt_on_glacier, rain_on_glacier]):
        var_name = vars[k]
        pf = pfs[k]
        #create netcdf file for each variable (maybe only do this for melt_on, liq_prcp_on because we do not need the others)
        name = '{}_on'.format(var_name)
        namefile = outpath + name+ '_' + out_name + '.nc'
        add_functions.create_netcdf(namefile, name, lat, lon, timeseries, "m3/d", resolution)

        #now total melt input for timeseries
        #output is in kg/day -> transform to m3/day by dividing by 1000
        #TODO: why is there a divison by pf??? this should only be thee case for liquid precipitation??
        glacier_flux_on = flux_on_glacier[start_index:end_index+1]
        glacier_flux_on = glacier_flux_on / 1000 / float(pf)

        ds = nc.Dataset(namefile, 'a')

        for i in range(len(timeseries)):
        #for i in range(40):
            raster = np.zeros((cellnr_lat, cellnr_lon))
            rasterflat = raster.flatten(order="F")

            for j,rgi in enumerate(glacier_ids):
                xy = glacier_dict[rgi][1]
                rasterflat[xy] += glacier_flux_on[i,j]
                ii =1
            rasterflat[rasterflat < 1e-9] = 0.0
            raster = np.reshape(rasterflat, (cellnr_lat, cellnr_lon),order="F")
            #var_nc[i, :, :] = raster
            ds.variables[name][i, :, :] = raster

        ds.close()
        del ds
    ii = 1





def oggm_area_to_cwatm_input(glacier_area_csv, oggm_results_org, cell_area, outpath, out_name, example_netcdf, resolution, fraction=True, fixed_year=None, include_off_area = False):
    '''
    This function generates a netcdf file of the area (area fraction) covered by glacier in each gridcell.
    Note that only the glaciers which were run by OGGM will be taken into account, which are normally the glaciers that drain into the basin at the model resolution.
    Note that only the area of the glacier within the gridded basin outline will be used, so that the total area is likely lower than the total area in OGGM results.

        :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
        :param oggm_results: results from oggm run, can either be
        :param cell_area: netcdf with global cell_area at output resolution
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :param example_netcdf: example netcdf with same extent that you want to generate
        :param fraction: if netcdf should contain area fraction of gridcell covered by glacier
        :param fixed_year: if also netcdf file for a fixed year should be generated
        :param include_off_area: whether to use constant area in OGGM or variable area

        returns: netcdf file of glacier areas per year
    '''
    #if outpath does not exist make it
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    if isinstance(oggm_results_org, list):
        oggm_results = []
        for oggm_results_org_part in oggm_results_org:
            #drop rgi_ids with nan values
            oggm_results_part = oggm_results_org_part.dropna('rgi_id', thresh=10)
            #check if OGGM results contains glaciers with missing results
            if len(oggm_results_org_part.rgi_id) != len(oggm_results_part.rgi_id):
                #get RGI IDs of missing values
                rgi_ids_nan = list(set(list(oggm_results_org_part.rgi_id.values)) - set(list(oggm_results_part.rgi_id.values)))
                #get number of glaciers with nan values
                missing_rgis = len(oggm_results_org_part.rgi_id) - len(oggm_results_part.rgi_id)
                #print warning that netcdf will not be generated for the glaciers with missing OGGM results
                msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(missing_rgis, rgi_ids_nan)
                warnings.warn(msg)
            oggm_results.append(oggm_results_part)

    else:
        # drop rgi_ids with nan values
        oggm_results = oggm_results_org.dropna('rgi_id', thresh=10)
        # check if OGGM results contains glaciers with missing results
        if len(oggm_results_org.rgi_id) != len(oggm_results.rgi_id):
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results_org.rgi_id.values)) - set(list(oggm_results.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results_org.rgi_id) - len(oggm_results.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
        oggm_results = [oggm_results]

    # define extent for which glacier maps should be created
    #if an example netcdf is given
    if type(example_netcdf) == str:
        # open it
        example_nc = xr.open_dataset(example_netcdf)
        #  get the coordinates, the cell width and the number of cells
        lat = np.round(example_nc.lat.values,
                       decimals=5)  # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = np.round(example_nc.lon.values,
                       decimals=5) # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth = np.round((lat[0] - lat[-1]) / (len(lat)-1), decimals = 5) #lat[0] -lat[1]
        cellnr_lat = len(lat)
        cellnr_lon = len(lon)
    #if you do not have an example netcdf file you need the cellsize, the number of cells and the coordinates in upper left corner
    elif len(example_netcdf) == 5:
        lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth = example_netcdf
        lon_max = lon_min + (cellnr_lon * cellwidth)
        lat_min = lat_max - (cellnr_lat * cellwidth)
        lon = np.round(np.arange(lon_min + cellwidth/2, lon_max, cellwidth), decimals=5)
        # #start latitude from -60 because there is only ocean below anyways
        lat = np.round(np.arange(lat_min + cellwidth/2, lat_max, cellwidth), decimals=5)
        cellnr_lon = np.round(cellnr_lon, decimals=5)
        cellnr_lat = np.round(cellnr_lat, decimals=5)
        cellwidth = np.round(cellwidth, decimals=5)
    # if neither of it is giving, the function does not work
    else:
        msg = 'The input {} is not a valid input. EIther give path to an example netcdf file or provide ' \
              'lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth of the upper left corner of model domain'.format(example_nc)
        raise ValueError(msg)

    x = np.zeros((cellnr_lat, cellnr_lon))
    glacier_on_area_array = x
    glacier_on_area_array = glacier_on_area_array.flatten()

    #define mask attributes from example_netcdf
    mask_attributes = [np.min(lon), np.max(lat), cellwidth, cellnr_lon, cellnr_lat]

    #get area of each gridcell covered by glacier for the timeperiod of OGGM run
    # check if cell_area and example_nc have the same cellwidth (same resolution)
    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1/12
    np.testing.assert_almost_equal(cellwidth, cellwidth_res, decimal=4,
                                   err_msg='example_nc and resolution need to have same resolution', verbose=True)
    if fraction == True:
        #check if cell_area and example_nc have the same cellwidth (same resolution)
        np.testing.assert_almost_equal(cellwidth, abs(cell_area.lon[1].values -cell_area.lon[0].values), decimal=4, err_msg='example_nc and cell_area need to have same resolution', verbose=True)
        area_gl_gridcell = change_area(glacier_area_csv, oggm_results, mask_attributes, include_off_area = include_off_area, cell_area=cell_area)
    else:
        area_gl_gridcell = change_area(glacier_area_csv, oggm_results, mask_attributes, include_off_area = include_off_area)

    timeseries = pd.date_range(str(area_gl_gridcell.columns.values[0]), str(area_gl_gridcell.columns.values[-1]),freq='AS')

    assert len(timeseries) == np.shape(area_gl_gridcell)[1]


    #create netcdf file for variable
    if include_off_area:
        label = "total_area"
    else:
        label = "on_area"
    if fraction == True:
        ds = nc.Dataset(outpath + label + '_fraction_' + out_name +'.nc', 'w', format='NETCDF4')
    else:
        ds = nc.Dataset(outpath + label + '_' + out_name + '.nc', 'w', format='NETCDF4')
    # add dimenstions, specify how long they are
    # use 0.5° grid
    #dimensions are created based on example netcdf
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', None)
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"
    #create variables glacier area
    var_nc = ds.createVariable(label, 'f4', ('time', 'lat', 'lon',),zlib=True)
    if fraction:
        var_nc.units = "fraction of cell_area"
    else:
        var_nc.units = "m2" #units is m2

    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)
    timeseries_str = str(timeseries)
    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    glacier_ids = area_gl_gridcell.index

    for i in range(len(timeseries)):
        glacier_on_area_array[glacier_ids] = area_gl_gridcell.iloc[:,i].values
        # glacier_on_area[glacier_ids] = np.ones(len(glacier_ids))
        glacier_on_area_2d = np.reshape(glacier_on_area_array, (cellnr_lat, cellnr_lon))
        var_nc[i, :, :] = glacier_on_area_2d
    ds.close()

    if fixed_year:
        if fraction == True:
            ds_fixed = nc.Dataset(outpath + label + '_fraction_' + out_name + '_constant_' + str(fixed_year) + '.nc',
                                  'w', format='NETCDF4')
        else:
            ds_fixed = nc.Dataset(outpath + label + '_' + out_name + '_constant_' + str(fixed_year) + '.nc', 'w',
                                  format='NETCDF4')

        lat_dim_fixed = ds_fixed.createDimension('lat', len(lat))
        lon_dim_fixed = ds_fixed.createDimension('lon', len(lon))
        lats_fixed = ds_fixed.createVariable('lat', 'f4', ('lat',))
        lons_fixed = ds_fixed.createVariable('lon', 'f4', ('lon',))
        # create variables glacier melt on off, liquid precipitation on off
        var_fixed_nc = ds_fixed.createVariable(label, 'f4', ('lat', 'lon',),zlib=True)
        if fraction:
            var_fixed_nc.units = "fraction of cell_area"
        else:
            var_fixed_nc.units = "m2"  # units is m2

        lats_fixed[:] = np.sort(lat)[::-1]
        lons_fixed[:] = np.sort(lon)

        x = np.zeros((cellnr_lat, cellnr_lon))
        glacier_on_area = x
        glacier_on_area = glacier_on_area.flatten()
        glacier_ids = area_gl_gridcell.index

        glacier_on_area[glacier_ids] = area_gl_gridcell.iloc[:,np.argwhere(area_gl_gridcell.columns == fixed_year)[0][0]].values
        glacier_on_area_2d = np.reshape(glacier_on_area, (cellnr_lat, cellnr_lon))
        var_fixed_nc[:, :] = glacier_on_area_2d

        ds_fixed.close()



def oggm_area_to_cwatm_input_world_PB(rgi_ids,glacier_dict,grid_dict,  oggm_results_path, startyear, endyear, outpath, out_name, example_netcdf,resolution, fraction=True, fixed_year=None):

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # define extent for which glacier maps should be created
    lat,lon, cellwidth, cellnr_lat,cellnr_lon, cellwidth_res = add_functions.read_example_nc(example_netcdf, resolution)

    glacier_on_area_array = np.zeros((cellnr_lat, cellnr_lon))
    glacier_on_area_array = glacier_on_area_array.flatten()

    # define mask attributes from example_netcdf
    mask_attributes = [lat,lon, np.min(lon), np.max(lat), cellwidth, cellnr_lon, cellnr_lat]

    #if fraction == True:
    change_area_world_PB(rgi_ids,glacier_dict,grid_dict, oggm_results_path, mask_attributes, startyear, endyear, outpath + out_name,resolution)
    #else:
    #area_gl_gridcell = change_area_world(glacier_area_csv, oggm_results_path, mask_attributes, startyear, endyear, outpath,include_off_area=include_off_area)

    #timeseries = pd.date_range(str(area_gl_gridcell.columns.values[0]), str(area_gl_gridcell.columns.values[-1]),
    #                           freq='AS')
    #assert len(timeseries) == np.shape(area_gl_gridcell)[1]
    ii = 1



def change_area(glacier_area_csv, oggm_results_list, mask_attributes, include_off_area=False, cell_area = None):
    '''
    Update glacier area in each gridcell by substracting the decreased area from all gridcells the glacier is covering, relative to the percentage coverage
    function works for area reduction and area growth

    :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
    :param oggm_results: results from oggm run
    :return: a pandas data frame of the glacier area in each grid cell partially covered by glaciers
            with index tuple of lat, lon of gridcell and
            with columns all years in oggm_results and their corresponding area
    '''
    min_lon, max_lat, cellwidth, cellnr_lon, cellnr_lat = mask_attributes

    for i, oggm_results in enumerate(oggm_results_list):

        #TODO: is there a faster option than using pandas?
        #get IDs of glaciers modelled by OGGM
        rgi_ids = oggm_results.rgi_id.values
        #only look at glaciers which were modelled by OGGM
        #TODO: should this be for all OGGM_results together?
        glacier_area_basin = glacier_area_csv[np.isin(glacier_area_csv.RGIId, rgi_ids)]
        glacier_area_basin = glacier_area_basin.reset_index(drop=True)
        #make a new array that contains latitudes longitudes, Gridcell Nr and years of data corresponding to length of OGGM results
        # + 2 because we need lat, lon, Nr Gridcells but we do not need last year
        array_area = np.zeros((np.shape(glacier_area_basin)[0], np.shape(oggm_results.time)[0] + 2))
        array_area[:, 0] = glacier_area_basin.Nr_Gridcell.values
        array_area[:,1] = glacier_area_basin.Latitude.values
        array_area[:,2] = glacier_area_basin.Longitude.values

        x = np.zeros((cellnr_lat, cellnr_lon))
        x = x.astype(int)
        glacier_on_area_array = x
        glacier_on_area_array = glacier_on_area_array.flatten()

        #TODO loop through rgi results if necessary
        #loop through all glaciers to reduce area of each glacier in the gridcells that are covered by the glacier
        for rgi_id in rgi_ids:
            #get all rows of df of current glacier
            current_glacier = glacier_area_basin[glacier_area_basin.RGIId == rgi_id]
            #area of RGI is start area, because for this area we have the outlines
            area_start = np.sum(glacier_area_basin[glacier_area_basin.RGIId == rgi_id].Area)
            #area of all years from OGGM results
            if include_off_area:
                area_oggm = oggm_results.off_area.loc[:, rgi_id].values + oggm_results.on_area.loc[:, rgi_id].values
            else:
                area_oggm = oggm_results.on_area.loc[:, rgi_id].values
            #area reduction is relative to area at RGI date
            #TODO: area reduction should be relative to area at start date
            area_reduction = area_start - area_oggm
            #get relative area reduction compared to glacier area at RGI date
            rel_reduction = area_reduction / area_start
            #multiply the relative area that remains with the glacier area in the grid cell
            area_glacier = np.outer((1 - rel_reduction), current_glacier.Area)
            #area that remains should be the same as glacier area in oggm
            #if area close to zero, relative differences can be large, therefore atol = 0.001

            np.testing.assert_allclose(np.sum(area_glacier, axis=1), area_oggm, rtol=1e-3, atol = 0.1)# for past: rtol=1e-5, atol=0.001)
            # everything below a an area of 1 should be neglected to avoid ridiculously small areas
            area_glacier = np.where(area_glacier < 1, 0, area_glacier)

            #put result in array to generate dataframe
            assert np.all(current_glacier.Latitude.values == array_area[current_glacier.index.values, 1])
            assert np.all(current_glacier.Longitude.values == array_area[current_glacier.index.values, 2])
            array_area[current_glacier.index.values, 3:] = area_glacier[:-1, :].T

        #generate a dataaframe
        if i == 0:
            #TODO: here it should be appended
            df = pd.DataFrame(array_area,
                              columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(oggm_results.time.values.astype('str')[:-1]))
        else:
            df = pd.concat([df, pd.DataFrame(array_area,
                              columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(oggm_results.time.values.astype('str')[:-1]))])

        print(np.shape(df))
    #TODO: this assertion does not always work only if RGI date it 2000?
    #for small glacier area rel_dif can be large
    #for large glacier area abs_diff can be large
    # if oggm_results.time.values[0] < 2000:
    #     np.testing.assert_allclose(df.iloc[:, 18], glacier_area_basin.Area, rtol=0.02, atol = 50000)
    #sum area across glaciers for same gridcells by using the Nr Gridcell as indicator
    dfinal = pd.DataFrame(df.groupby(by='Nr_Gridcell').sum().iloc[:, 2:].values,
        columns=list(oggm_results.time.values.astype('int')[:-1]))
    #get the latitude, longitude of these gridcells by taking mean ofver Nr of Gridcells
    round_lat = np.round(df.groupby(by='Nr_Gridcell').mean().Latitude, decimals=3)
    round_lon = np.round(df.groupby(by='Nr_Gridcell').mean().Longitude, decimals=3)

    cell_lat = (max_lat - np.array(round_lat)) / cellwidth
    cell_lon = (np.array(round_lon) - min_lon) / cellwidth
    #make sure that cell_lat and cell_lon are indices (should be integer)
    if cell_lat.all() % 1 < 0.1 or cell_lat.all() % 1 > 0.9:
        cell_lat = np.round(cell_lat).astype('int')
    else:
        print(cell_lat)
        raise ValueError
    if cell_lon.all() % 1 < 0.1 or cell_lon.all() % 1 > 0.9:
        cell_lon = np.round(cell_lon).astype('int')
    else:
        print(cell_lon)
        raise ValueError

    #celllat and cell_lon should be within the bounds of the example_nc
    #crop all entries that are not within the mask attributes
    #arg_out = np.argwhere((cell_lat < 0) | (cell_lat >= cellnr_lat) | (cell_lon < 0) | (cell_lon >= cellnr_lon)).flatten()
    arg_in =np.argwhere((cell_lat >= 0) & (cell_lat < cellnr_lat) & (cell_lon >= 0) & (cell_lon < cellnr_lon)).flatten()
    cell_lat = cell_lat[arg_in]
    cell_lon = cell_lon[arg_in]
    #crop values where index is larger than index
    if len(arg_in) < len(cell_lat):
        warnings.warn(
            "The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")
    dfinal_array = np.array(dfinal)

    #set the index of the dataframe as a tuple of latitude, longitude
    #TODO: in case you want to have fraction of area instead of total area divide area by area fraction
    if cell_area:
        ind_lat_area = [np.argmin(abs(cell_area["lat"].values - x)) for x in round_lat] #np.argmin(abs(cell_area["lat"].values - round_lat))
        ind_lon_area = np.array([np.argmin(abs(cell_area["lon"].values - x)) for x in round_lon]) #np.argmin(abs(cell_area["lon"].values - round_lon))
        assert len(ind_lon_area) == len(ind_lat_area)
        cell_area_gl =[]
        for k in range(len(ind_lon_area)):
            cell_area_gl.append(cell_area[list(cell_area.keys())[0]].values[ind_lat_area[k]][ind_lon_area[k]])
        # only add area if grid cell is on land
        #TODO only add area if grid cells are on land
        #dfinal = dfinal.iloc[:, :].values / np.array(cell_area_gl)[:, None]
        dfinal_array = np.divide(dfinal.iloc[:, :].values, np.array(cell_area_gl)[:, None], where=np.array(cell_area_gl)[:, None] != 0,
                  out=np.zeros(np.shape(dfinal.iloc[:, :].values)))

    dfinal_array = dfinal_array[arg_in, :]

    dfinal = pd.DataFrame(dfinal_array,
                          columns=list(oggm_results.time.values.astype('int')[:-1]))
    #dfinal.index = list(zip(round_lat, round_lon))

    ind_cell = (cell_lat) * cellnr_lon + cell_lon
    dfinal.index = ind_cell

    return dfinal


def change_area_world_PB(rgi_dict, glacier_dict, grid_dict, oggm_results_list, mask_attributes, startyear, endyear, outname, resolution):

    lat,lon,min_lon, max_lat, cellwidth, cellnr_lon, cellnr_lat = mask_attributes
    for i, oggm_results_path in enumerate(oggm_results_list):
        print(oggm_results_path)

        oggm_results = xr.open_dataset(oggm_results_path)
        #delete nan values
        nok = oggm_results.volume.isel(time=0).isnull()
        # check if OGGM results contains glaciers with missing results
        if np.count_nonzero(nok) > 0: #len(oggm_results_org.rgi_id) != len(oggm_results.rgi_id):
            oggm_results_new = oggm_results.dropna('rgi_id', thresh=10)
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results.rgi_id.values)) - set(list(oggm_results_new.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results.rgi_id) - len(oggm_results_new.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
            oggm_results.close()
            del oggm_results
            oggm_results = oggm_results_new
            del oggm_results_new

        # select only those one in a list
        if not(rgi_dict == None):
            oggm_results = oggm_results.sel(rgi_id=rgi_dict)

        years_oggm = list(oggm_results.time.values.astype('int')[:-1])
        years = list(np.arange(startyear, endyear + 1))
        assert all(item in years_oggm for item in years), 'Years are out of range of years of OGGM results. Change startyear and endyear.'

        #get IDs of glaciers modelled by OGGM
        rgi_ids = oggm_results.rgi_id.values

        #get glacier areas as modeled by OGGM (for x years)
        area_oggm = [oggm_results.on_area.loc[years, rgi_id].values for rgi_id in rgi_ids]
        area_start = [np.sum(glacier_dict[rgi][3]) for rgi in rgi_ids]


        #test
        tt = pd.DataFrame(area_oggm)
        tt.index = rgi_ids
        tt.columns=years
        tt.to_csv("P:/watmodel/CWATM/Regions/CWatM-Otta/glaciers/glacier_area_from_oggmresult.csv")



        #calculate area change (can be reduction or growth
        rel_change = list(map(lambda x, y: 1-y/x, area_start, area_oggm))
        # calculate area of glacier using area change
        area_glacier = list(map(lambda x, y: (np.outer((1 - x), y)), rel_change, area_start))

        # delete current oggm result from workspace
        oggm_results.close()
        del oggm_results

        timeseries = pd.date_range("1990", str(endyear), freq="YS")
        add_functions.create_netcdf(outname+".nc", "on_area", lat, lon, timeseries, "-", resolution)
        #create_netcdf2(outname,"area_fraction", example_nc, timeseries, "-")
        nf1 = nc.Dataset(outname+".nc", 'a')

        # depending on the elevation rank of the grid a function value is used
        # to add or remove ice from the glacier -> the higher up, the less ice is removed
        # function for splitting change into different cells
        splitF = [x for x in range(1,101)]
        #splitF = [x*x for x in range(1, 101)]
        # put in raster


        # test2
        test2 = np.zeros((len(glacier_dict), endyear-startyear+1))

        for j,year in enumerate(years):
            rasterflat = np.zeros((cellnr_lat, cellnr_lon)).astype(float).flatten(order="F")

            for i,rgi in enumerate(rgi_ids):
                #year = 1990
                #if year == 2007:

                if rgi == "RGI60-01.00003":  #2769 AS SINGLE GLAcier point
                        iii = 1
                #j = 0
                #i =3
                #i =0
                #rgi = "RGI60-08.00312"
                #rgi = "RGI60-08.01192"

                # number of gridcells covered by glacier
                no_cells = len(glacier_dict[rgi][0])
                # total cell area : in cased of 1km it is always the same
                total_cellarea = list(map(lambda x: grid_dict[x][2], glacier_dict[rgi][0]))
                # split change by number of cells and function
                split = np.add.reduceat(splitF, range(0, (100//no_cells)*no_cells, 100//no_cells))
                split = split / np.sum(split)

                oldarea = glacier_dict[rgi][3]
                oldperc = oldarea / total_cellarea
                newarea = oldarea.copy()
                newperc = newarea / total_cellarea

                if j == 0:
                    # glacier area the year before as starting point (or the initial value)
                    area_ini = area_start[i]
                else:
                    area_ini = np.sum(oldarea)

                area = np.sum(area_glacier[i][j])

                # get the absolute chnage in glacier area, multiply with change function
                # and add/substract it f4rom old area
                # doesnt matter if glacier area become negative -> that's looked after next
                change = area-area_ini  # negative is smaller size of glacier
                changepercell = change * split
                #oldarea = glacier_dict[rgi][3]
                #oldperc = oldarea / total_cellarea
                #newarea = oldarea.copy()

                # add/subs from top elevation to down
                """
                We run the glacier grid points 2 times:
                1 ) from top elevation to low elevation -> removing /adding on top elevsation first - in case it is empty/full remove/add from cell at lower elevation
                2)
                """
                addarea  = 0
                for k,grid in enumerate(glacier_dict[rgi][0]):
                    newarea[k] += changepercell[k] + addarea
                    addarea = 0
                    newperc[k] = newarea[k] / total_cellarea[k]

                    # test if cell would fall < 0 -> glacier is shrinking
                    if newarea[k] < 0:
                        if (k+1) < no_cells:
                            # add part which is below 0 on the next elevation, but cannot go further then 1 above lowest gridcell
                            addarea = newarea[k]  # newarea is always negative here
                            newarea[k] = 0
                            newperc[k] = 0.
                        else:
                            ii = 1
                            # just a dummy, if area is < in the lowest cell this is will be worked on in the next loop

                    """
                    # do not put holes into glacier areas
                    # test if cell is 100% glacier and next cell too -> than another
                    if (k+2) < no_cells:
                        # if glacier has 100% and the next one too, than do not take area from this, because it looks strange
                        maxperc1 = 0.999999 - (grid_dict[grid][1] - oldarea[k] / grid_dict[grid][2])
                        maxperc2 = 0.999999 - (grid_dict[glacier_dict[rgi][0][k+1]] - oldarea[k] / grid_dict[glacier_dict[rgi][0][k+1]][2])
                        if newarea[k]  > maxperc1:
                    """



                    # test if cell has too much glacier
                    maxperc = 1.0 - (grid_dict[grid][1] - oldperc[k])
                    # if it is the lowest cell look for expansion
                    if k == no_cells:
                        expandglacier = 0.5
                        if newperc[k] > expandglacier * maxperc:  # look for glacier expansion if cell is half full
                            addarea = newperc[k] - expandglacier * maxperc
                            newperc[k] = expandglacier * maxperc
                            ldd = glacier_dict[rgi][5][k]
                            if ldd in grid_dict.keys():
                                max1 = (1 - grid_dict[ldd][1]) * grid_dict[ldd][2]
                                # if there is space than put the load
                                if max1 > 0:
                                    # check if there is enough space
                                    if addarea < max1:
                                        addarea1 = addarea
                                    else:
                                        addarea1 = max1
                                    # TODO what happens to the snow which doesnt fit in: addarea - max1
                                    # test if cell is already in glacier
                                    id = np.where(glacier_dict[rgi][0] == ldd)
                                    if not (id[0]):
                                        # cell does not have this glacier -> add to glacier and to gridcell dict.
                                        glacier_dict[rgi][0] = np.append(glacier_dict[rgi][0], ldd)
                                        glacier_dict[rgi][2] = np.append(glacier_dict[rgi][2],
                                                                         addarea1 / grid_dict[ldd][2])
                                        glacier_dict[rgi][3] = np.append(glacier_dict[rgi][3], addarea1)
                                        glacier_dict[rgi][4] = np.append(glacier_dict[rgi][4], 0)  # dem
                                        glacier_dict[rgi][5] = np.append(glacier_dict[rgi][5], 0)  # ldd
                                        grid_dict[ldd][0].append(rgi)

                                    else:
                                        # glacier already has this cell -> so only add the additional area
                                        glacier_dict[rgi][3][id] = glacier_dict[rgi][3][id] + addarea1
                                        glacier_dict[rgi][2][id] = glacier_dict[rgi][3][id] / grid_dict[ldd][2]
                                    grid_dict[ldd][1] += addarea1 / grid_dict[ldd][2]
                                    # change total percentage of grid glacier area
                                    iii = 1
                            else:
                                iii =1




                    # maxperc is the max the glacier can expand because other glacier may occupy this cell too
                    if newperc[k] > (maxperc + 0.000001):
                        addarea = (newperc[k] - maxperc) * total_cellarea[k]
                        # new percentage is maxpertage (could be < 1 because of other glaciers in the same gridcell
                        newperc[k] = maxperc

                        # if lowest cell is reached it checks for additional area
                    else:
                        iii = 1

                # ------------------------------
                # Second round - from low to high
                # now it checks if glacier area is too less (<0) or too big (> 1 - other glaciers in the grid)
                for k, grid in reversed(list(enumerate(glacier_dict[rgi][0]))):
                    # loop backwards from low elevation to high

                    # test if glacier area become < 0
                    #-------------------------------
                    if newarea[k] < 0:
                        # test if newarea is smaller than 0 -> need to distribute
                        add = newarea[k]
                        # add is subtracted from higher elevation cell
                        newarea[k] = 0.
                        newperc[k] = 0.
                        if k > 0:
                            # only to highest elevation, otherwise glacier area is totally lost
                            # higher elevation cell gets the amount of ice substracted which cannot sub. from lower cell
                            newarea[k-1]  += add  # add is always negative here
                            newperc[k-1] = newarea[k-1] / total_cellarea[k-1]
                        else:
                            # sometime high elevation has small glacier area too
                            print ("bye bye glacier: ", rgi,year,grid)
                    # -----------end glacier shrink--------------------

                    # if glacier area becomes to big
                    # try to find upstream to remove glacier
                    # --------------------------------------------
                    maxperc = 1.0 - (grid_dict[grid][1] - oldperc[k])
                    # maxperc is the max the glacier can expand because other glacier may occupy this cell too

                    if newperc[k] > (maxperc + 0.000001):
                        # if it is only little bit bigger ignore
                        # additional amount to put somewhere else
                        addarea = (newperc[k] - maxperc) * total_cellarea[k]
                        # new percentage is maxpertage (could be < 1 because of other glaciers in the same gridcell
                        newperc[k] = maxperc

                    # --------------- end glacier grow -----------------------------
                    if grid == 12067:
                        iii =1
                    rasterflat[grid] += newperc[k]
                # end loop glacier
                glacier_dict[rgi][3] = newperc * total_cellarea

                test2[i,j] = np.sum( glacier_dict[rgi][3])
                iii =1
            # end loop year checking if cell must be redistributed

            rasterflat[rasterflat < 1e-9] = 0.0
            raster = np.reshape(rasterflat, (cellnr_lat, cellnr_lon), order="F")

            nf1.variables["on_area"][j,:, :] = raster

            id = glacier_dict[rgi_ids[0]][0]
            a = np.sum(rasterflat[id])
        nf1.close()

        #test
        tt = pd.DataFrame(test2)
        tt.index = rgi_ids
        tt.columns=years
        tt.to_csv("P:/watmodel/CWATM/Regions/CWatM-Otta/glaciers/glacier_area_from_grid.csv")

        #rasterflat[rasterflat < 1e-9] = 0.0
        #raster = np.reshape(rasterflat, (cellnr_lat, cellnr_lon), order="F")
        #var_nc[i, :, :] = raster

        ii =1













