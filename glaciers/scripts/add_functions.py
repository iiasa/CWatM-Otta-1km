import netCDF4 as nc
import xarray as xr
import numpy as np

def read_example_nc(example_netcdf, resolution):
    example_nc = xr.open_dataset(example_netcdf)

    if resolution == '1km':
        lat = example_nc.y.values # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = example_nc.x.values # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth_res =  1000.

    else:
        # danube lon 8-30 lat 41-51
        lat = example_nc.lat.values[468:588]    # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = example_nc.lon.values[2256:2520]   # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
    cellwidth = np.round((lat[0] - lat[-1]) / (len(lat) - 1), decimals=5)  # lat[0] -lat[1]
    cellnr_lat = len(lat)
    cellnr_lon = len(lon)

    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1 / 12

    return lat,lon, cellwidth, cellnr_lat,cellnr_lon, cellwidth_res

def create_netcdf(name,varname,lat,lon,timeseries,unit,resolution):

    ds = nc.Dataset(name, 'w', format='NETCDF4_CLASSIC')

    ds.keywords ='CWATM, OGGM, Danube, 5armin'
    ds.source = 'CWATM maps for Danube based on glacier model OGGM'
    ds.institution ="IIASA Water Security"

    # add dimenstions, specify how long they are
    # use 0.5° grid
    if resolution == "1km":
        lat_dim = ds.createDimension('y', len(lat))
        lon_dim = ds.createDimension('x', len(lon))
        lats = ds.createVariable('y', 'f8', ('y',))
        lats.standard_name = 'projection_y_coordinate'
        lats.long_name = 'y coordinate of projection'
        lats.units = 'Meter'
        lons = ds.createVariable('x', 'f8', ('x',))
        lons.standard_name = 'projection_x_coordinate'
        lons.long_name = 'x coordinate of projection'
        lons.units = 'Meter'

        #if resolution == "1km":
        prj = ds.createVariable('transverse_mercator', 'i4')
        prj.grid_mapping_name = 'transverse_mercator'
        prj.false_easting = 500000.0
        prj.false_northing = 0.0
        prj.longitude_of_central_meridian = 15.0
        prj.latitude_of_projection_origin = 0.0
        prj.scale_factor_at_central_meridian = 0.9996
        prj.long_name = 'CRS definition'
        prj.semi_major_axis = 6378137.0
        prj.inverse_flattening = 298.257222101
        prj.spatial_ref = '"PROJCS[\"ETRS89 / UTM zone 33N\",GEOGCS[\"ETRS89\",DATUM[\"European_Terrestrial_Reference_System_1989\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],AUTHORITY[\"EPSG\",\"6258\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4258\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",15],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"25833\"]]"'
        prj.EPSG_code = "EPSG:25833"


    else:
        lat_dim = ds.createDimension('lat', len(lat))
        lon_dim = ds.createDimension('lon', len(lon))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))

    time_dim = ds.createDimension('time', len(timeseries))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"

    # create variables glacier melt on off, liquid precipitation on off
    if resolution == "1km":
        var_nc = ds.createVariable(varname, 'f4', ('time', 'y', 'x',), chunksizes=(1, len(lat), len(lon)),
                               zlib=True)  # chunksizes=(1,len(lat),len(lon))
    else:
        var_nc = ds.createVariable(varname, 'f4', ('time', 'lat', 'lon',), chunksizes=(1, len(lat), len(lon)),
                               zlib=True)  # chunksizes=(1,len(lat),len(lon))
    var_nc.units = unit
    if resolution == "1km":
        var_nc.standard_name = varname
        var_nc.long_name = varname
        var_nc.grid_mapping = 'transverse_mercator'
        var_nc.esri_pe_string = 'PROJCS["ETRS89 / UTM zone 33N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","25833"]]'


    # use the extent of the example netcdf
    # TODO maybe no example netcdf needed but it can be done from scratch
    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)
    timeseries_str = str(timeseries)
    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    ds.close()


















def create_netcdf3(name,varname,example_netcdf,timeseries,unit,resolution):

    example_nc = xr.open_dataset(example_netcdf)
    lat = example_nc.lat.values # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
    lon = example_nc.lon.values
    cellwidth = np.round((lat[0] - lat[-1]) / (len(lat)-1), decimals = 5) #lat[0] -lat[1]
    cellnr_lat = len(lat)
    cellnr_lon = len(lon)

    #create netcdf file for each variable (maybe only do this for melt_on, liq_prcp_on because we do not need the others)
    ds = nc.Dataset(name + '.nc', 'w', format='NETCDF4')
    # add dimenstions, specify how long they are
    # use 0.5° grid
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', len(timeseries))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"

    var_nc = ds.createVariable(varname, 'f4', ('time', 'lat', 'lon',), chunksizes=(1,len(lat),len(lon)), fill_value=1e20, zlib=True)
    var_nc.units = unit

    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)

    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    ds.close()
