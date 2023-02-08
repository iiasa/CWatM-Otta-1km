Soil data
----------
PB 27/10/2020

http://globalchange.bnu.edu.cn/research/soil5d.jsp
http://globalchange.bnu.edu.cn/research/soil5.jsp

A new version of the global high-resolution dataset of soil hydraulic and thermal parameters for land surface modeling
!!NOTE: Please limit the number of download threads under 5!
If too many threads are detected, the IP addresses may be blocked.
Wget command is a good choice for downloading.

Introduction | Data citation | Data download
Introduction

A newly developed global dataset of soil hydraulic and thermal parameters using multiple Pedotransfer Functions (PTFs) that are widely cited or recently developed is provided for Land Surface Modeling.

The dataset consists of two sets of parameters derived respectively from the Global Soil Dataset for Earth System Models (GSDE) [Shangguan et al., 2014] and SoilGrids [Hengl et al., 2014, 2017] databases. The published variables are listed in the table below. These variables are all provided at the spatial resolution of 30" ranging from 90oN to 90oS, 180oW to 180oE, with four sets of vertical profiles available [i.e., as vertical resolutions of SoilGrids (0 - 0.05 m, 0.05 - 0.15 m, 0.15 - 0.30 m, 0.30 - 0.60 m, 0.60 - 1.00 m, and 1.00 - 2.00 m), Noah-LSM (0 - 0.1 m, 0.1 - 0.4 m, 0.4 - 1.0 m, and 1.0 - 2.0 m), JULES (0 - 0.1 m, 0.1 - 0.35 m, 0.35 - 1.0 m, and 1.0 - 3.0 m) and CoLM/CLM (0 - 0.0451 m, 0.0451 - 0.0906 m, 0.0906 - 0.1655 m, 0.1655 - 0.2891 m, 0.2891 - 0.4929 m, 0.4929 - 0.8289 m, 0.8289 - 1.3828 m, 1.3828 - 3.8019 m)]. The dataset is currently stored in the binary format.

The soil water retention parameters, based on the Campbell [1974] and van Genuchten [1980] (hereafter VG) models, are obtained from a fitting method to find the optimal water retention parameters from ensemble PTFs. The soil hydraulic conductivity is estimated as the median values of ensemble PTFs. The heat capacity of soil solids is calculated as the volumetric weighted average of the heat capacity of mineral soils, soil organic matters (SOM) and gravels. And the soil thermal conductivity is estimated following the models of Johansen [1975] and Balland and Arp [2005], with all the effects of soil constituents such as SOM and gravels considered.

A more detailed introduction to the dataset can be downloaded here.
Data citation

Dai, Y., N. Wei, H. Yuan, S. Zhang, W. Shangguan, S. Liu, and X. Lu (2019a), Evaluation of soil thermal conductivity schemes for use in land surface modelling, J. Adv. Model. Earth System, accepted.

Dai, Y., Q. Xin, N. Wei, Y. Zhang, W. Shangguan, H. Yuan, S. Zhang, S. Liu, and X. Lu (2019b), A global high-resolution dataset of soil hydraulic and thermal properties for land surface modeling, J. Adv. Model. Earth System, accepted.
Data download

The parameters derived from the GSDE can be downloaded currently in the following table, and those from SoilGrids will be published soon after the space for data storage is enlarged on our server.

---------------

8. 	Log-10 transformation of saturated hydraulic conductivity 	cm day-1	log10-KS	100
7. 	Saturated water content 	cm3 cm-3					Theta_s		0.5
11. 	Residual moisture content for the VG model 	cm3 cm-3			Theta_r		0.1
12. 	The inverse of the air-entry value for the VG model 	cm-1			Alpha		0.075

13. 	Log-10 transformation of a shape parameter for the VG model 	-		log10-n		0.23
10. 	Pore size distribution index for the Campbell model 	-			lambda		0.3

----------------
SoilGrids
1: 0    - 0.05 m	1.
2: 0.05 - 0.15 m		
3: 0.15 - 0.30 m 	2.						
4: 0.30 - 0.60 m 	
5: 0.60 - 1.00 m 	3.
6: 1.00 - 2.00 m	

----------------
cd P/p/luc/watproject/Datasets/soil/33_Dai


python3 tif2netcdf.py
cdo remapbil,elvstd.nc ks_l1.nc ks_30min.nc

do sellonlatbox,-35.5,74,23.9,73 ks_l1.nc ks1_eu1.nc



import arcpy
arcpy.env.workspace = "P:/watproject/Datasets/soil/33_Dai/ks"
##Reproject a TIFF image with Datumn transfer
arcpy.ProjectRaster_management("image.tif", "reproject.tif", "World_Mercator.prj",\
                               "BILINEAR", "5", "NAD_1983_To_WGS_1984_5", "#", "#")
                               

# Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script
# The following inputs are layers or table views: "ksat1"
arcpy.ProjectRaster_management(in_raster="ksat1", out_raster="P:/watproject/Datasets/soil/33_Dai/ks/ks1_eu4.tif", out_coor_system="PROJCS['ETRS_1989_LAEA',GEOGCS['GCS_ETRS_1989',DATUM['D_ETRS_1989',SPHEROID['GRS_1980',6378137.0,298.257222101]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Azimuthal_Equal_Area'],PARAMETER['False_Easting',4321000.0],PARAMETER['False_Northing',3210000.0],PARAMETER['Central_Meridian',10.0],PARAMETER['Latitude_Of_Origin',52.0],UNIT['Meter',1.0]]", resampling_type="BILINEAR", cell_size="1000 1000", geographic_transform="ETRS_1989_To_WGS_1984", Registration_Point="2500000 750000", in_coor_system="GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]]", vertical="NO_VERTICAL")


# Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script
# The following inputs are layers or table views: "ks1_eu4.tif", "mask_1km_2bit.tif"
arcpy.Clip_management(in_raster="ks1_eu4.tif", rectangle="2500000 750000 7500000 5500000", out_raster="P:/watproject/Datasets/soil/33_Dai/ks/ks1_eu6.tif", in_template_dataset="mask_1km_2bit.tif", nodata_value="1.000000e+20", clipping_geometry="NONE", maintain_clipping_extent="NO_MAINTAIN_EXTENT")
                               
                               
                               
##cdo sellonlatbox,2500000,7500000,750000,5500000 ks1_eu5.nc ks1_eu6.nc



---
theta_s
in some cases theta_s is lower than theta_r (even in the original maps)
therefore: range = ts-tr; if range <0.01,0.01, range, ts = tr+range


-----------------
bin2netcdf_alpha.py   writes .tif and netcdf of 30arcsec of original file - layer have to be set in py prg 
tif2eu.py  cuts to Europe and projects ETRS_1989_LAEA

----
from 30arcsec to 30arcmin
cdo remapbil,elvstd.nc ks_l1.nc ks_30min.nc

from 30arcsec to 5arcmin
cdo remapbil,dem.nc ks_l1.nc ks_5min.nc