#OTTA

#cdo seldate,2096-01-01,2098-12-31 otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_area_pf15_1km.nc  t1.nc
#cdo shifttime,+2years t1.nc t2.nc#
#cdo -f nc4 -z zip mergetime otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_area_pf15_1km.nc t2.nc otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_daily_2006_2100_area_pf15_1km.nc

cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2098_pf15_1km.nc t11.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2098_pf15_1km.nc t12.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_pf15_1km.nc t13.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2098_pf15_1km.nc t14.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2098_pf15_1km.nc t15.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_pf15_1km.nc t16.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2098_pf15_1km.nc t17.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2098_pf15_1km.nc t18.nc
cdo seldate,2096-01-01,2098-12-31 liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_pf15_1km.nc t19.nc

cdo shifttime,+2years t11.nc t21.nc
cdo shifttime,+2years t12.nc t22.nc
cdo shifttime,+2years t13.nc t23.nc
cdo shifttime,+2years t14.nc t24.nc
cdo shifttime,+2years t15.nc t25.nc
cdo shifttime,+2years t16.nc t26.nc
cdo shifttime,+2years t17.nc t27.nc
cdo shifttime,+2years t18.nc t28.nc
cdo shifttime,+2years t19.nc t29.nc

cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2098_pf15_1km.nc t21.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2098_pf15_1km.nc t22.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_pf15_1km.nc t23.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp26_r1i1p1_SMHI-RCA4_v1_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2098_pf15_1km.nc t24.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2098_pf15_1km.nc t25.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_pf15_1km.nc t26.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2098_pf15_1km.nc t27.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_DMI-HIRHAM5_v2_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2098_pf15_1km.nc t28.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_KNMI-RACMO22E_v2_daily_2006_2100_pf15_1km.nc
cdo -f nc4 -z zip mergetime liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_daily_2006_2098_pf15_1km.nc t29.nc liq_prcp_on_otta_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_daily_2006_2100_pf15_1km.nc
