#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: Jan 2022
Running offline version of dry deposition of GEOS-Chem on hourly timescale, adjusting for land type
@author: arifeinberg
"""
import xarray as xr
import numpy as np
import pandas as pd
from drydep_functions import Compute_Olson_landmap, METERO, DEPVEL
import scipy.io as sio
import datetime

def weird_division(n, d): # avoid dividing by 0
    return n / d if d else 0

#%% Load the observational dataset, used for finding coordinates to change land type

# Load dataset in pandas DataFrame
source = 'data/SI_Forest_Hg_uptake_database.csv'
data_dd_f = pd.read_csv(source)

# Only select stations for which a litterfall dry deposition velocity can be calculated
df_dd = data_dd_f.loc[~data_dd_f['Litterfall Dry Deposition velocity (cm s-1)'].isna()].reset_index()

# Load location information and observed land cover
obs_lat = df_dd['Latitude (N)'].values # latitude
obs_lon = df_dd['Longitude (E)'].values # longitude
obs_landcover = df_dd['Land Cover'].values # land cover type for measurements
location = df_dd['Location'].values # informative name for station
n_sites = len(obs_landcover) # number of sites

# Mapping name of land cover to dry deposition land types
# 1 - Snow/Ice 2 - Deciduous forest 3 - Coniferous forest 4 - Agricultural land
# 5 - Shrub/grassland 6 - Amazon forest 7 - Tundra 8 - Desert 9 - Wetland
# 10 - Urban 11 - Water
site_types = [None] * n_sites # initialize array
for i in range(n_sites):
    land_i = obs_landcover[i]
    if "Mixed" in land_i: # Mixed deciduous - coniferous
        site_types[i] = [2,3]
    elif ("Deciduous" in land_i) or ("Broadleaf" in land_i):
        site_types[i] = 2
    elif "Coniferous" in land_i:
        site_types[i] = 3
    elif "Rainforest" in land_i or "Ombrophilous" in land_i:
        site_types[i] = 6

#%% Load necessary variables for running dry deposition code

# Required Olson land variables, dry deposition parameters
fn_ols1 = 'data/Olson_2001_Drydep_Inputs.nc'
ds_ols1 = xr.open_dataset(fn_ols1)

DRYCOEFF = ds_ols1.DRYCOEFF.values # Baldocchi dry deposition polynomial coefficients
IOLSON = ds_ols1.IOLSON.values # Olson land type indices (+1)
IDRYDEP = ds_ols1.IDRYDEP.values # Dry deposition land types

IDEP = ds_ols1.IDEP.values # Mapping index: Olson land type ID to drydep ID
IZO = ds_ols1.IZO.values # Default roughness heights for each Olson land type
IRI = ds_ols1.IRI.values # RI resistance for each dry deposition land type
IRI[2] = 200. # Change RI resistance of coniferous forests to match deciduous, as in GEOS-Chem
IRLU = ds_ols1.IRLU.values # RLU resistance for each dry deposition land type
IRAC = ds_ols1.IRAC.values # RAC resistance for each dry deposition land type
IRGSS = ds_ols1.IRGSS.values # RGSS resistance for each dry deposition land type
IRGSO = ds_ols1.IRGSO.values # RGSO resistance for each dry deposition land type
IRCLS = ds_ols1.IRCLS.values # RCLS resistance for each dry deposition land type
IRCLO = ds_ols1.IRCLO.values # RCLO resistance for each dry deposition land type
IVSMAX = ds_ols1.IVSMAX.values # Max drydep velocity (for aerosol) for each dry deposition land type

# Load 2 x 2.5 map of land type areas
fn_ols2 = 'data/Olson_2001_Land_Type_Masks.2_25.generic.nc'
ds_ols2 = xr.open_dataset(fn_ols2)
# save as one np array, first dim is land types
Olson_landtype = ds_ols2.to_array().squeeze()

lon = ds_ols2.lon # longitude
lat = ds_ols2.lat # latitude

lon_l = len(lon)
lat_l = len(lat)
n_landtypes = 73 # number of Olson land types

# Species specific parameters
XMW = 201e-3 # Hg0 molar mass (kg/mol)
HSTAR = 0.11 # Hg0 Henry's Law Constant (M/atm)

F0_all = 3e-5 # Hg0 reactivity
F0_am = 0.2 # Hg0 reactivity in Amazon rainforest

#%% create daily interpolation for XLAI
# Land type LAI
fn_xlai  = 'Yuan_MODIS_XLAI_2_25_2015.nc' # weekly LAI data can be downloaded from GEOS-Chem WUSTL server
ds_xlai = xr.open_dataset(fn_xlai)

# save as one data array, first dim is land types
XLAI_w = ds_xlai.to_array()
temp = XLAI_w.reindex(time=pd.date_range("1/1/2015", "12/31/2015")) 
# As in GEOS-Chem, fill forwards, extrapolate first day by filling backwards
XLAI_d = temp.ffill("time").bfill("time").values 

#%% run dep velocity function

# Adjust the land use parameters

# initialize
DV_lt = np.zeros((n_sites, 12)) # array of deposition velocity monthly values for each site, corrected by land type

# Calculate information about Olson land types
FRCLND, IREG, ILAND, IUSE = Compute_Olson_landmap(Olson_landtype)
IREG_v = IREG.values
ILAND_v = ILAND.values
IUSE_v = IUSE.values

# Initialize new arrays for Olson Land types, for land correction
IREG_new = np.zeros(n_sites, dtype=int)
ILAND_new = np.zeros((n_landtypes, n_sites), dtype=int)
IUSE_new = np.zeros((n_landtypes, n_sites))

Olson_landtype_v = Olson_landtype.values

# loop through sites, adjust Olson land type parameters of each site

# keep in mind that one grid box may have different observation sites with
# different land types

# Find where Olson land types correspond to dry deposition land types:
# type 2 (deciduous), type 3 (coniferous),
# type 4 (agricultural), type 6 (rainforest)
IOLSON_2 = np.asarray(np.where(IDEP==2)).flatten()
IOLSON_3 = np.asarray(np.where(IDEP==3)).flatten()
IOLSON_4 = np.asarray(np.where(IDEP==4)).flatten()
IOLSON_6 = np.asarray(np.where(IDEP==6)).flatten()

# Adjust site types so that don't have to do this in loop
for i in range(n_sites):

    # find lon and lat indices of sites
    lat_i = np.argmin(np.abs(np.array(lat)-obs_lat[i]))
    lon_i = np.argmin(np.abs(np.array(lon)-obs_lon[i]))
        
    # adjust land cover according to observed type of land category
    if site_types[i] == 2:
        IREG_new[i]= 1
        # set only land type to 26, Broadleaf forest
        ILAND_new[0,i]= 26
        ILAND_new[1:,i]= -9999
        IUSE_new[0,i]= 1000.        
        IUSE_new[1:,i]= 0.       
    elif site_types[i] == 3:
        IREG_new[i]= 1
        # set only land type to 27, Conifer forest
        ILAND_new[0,i]= 27
        ILAND_new[1:,i]= -9999
        IUSE_new[0,i]= 1000.        
        IUSE_new[1:,i]= 0.        
    elif site_types[i] == [2, 3]: # Mixed forest
        IREG_new[i]= 2
        # set two land types, Conifer and Broadleaf forests
        ILAND_new[0:2,i]= [26,27]
        ILAND_new[2:,i]= -9999
        IUSE_new[0:2,i]= 500.        
        IUSE_new[2:,i]= 0.  
    elif site_types[i] == 6:
        IREG_new[i]= 1
        # set only land type to 33, Rainforest
        ILAND_new[0,i]= 33
        ILAND_new[1:,i]= -9999
        IUSE_new[0,i]= 1000.        
        IUSE_new[1:,i]= 0.  

# loop over months
for j in range(12): 
    print("j"+str(j))
    # load required met variables - can be downloaded from GEOS-Chem WUSTL server
    mth_s = '%02d' % (j+1) # string of month with leading zero if single digit
    fn_met = 'GEOSChem.StateMet.2015' + mth_s + '01_0000z.nc4'
    ds_met = xr.open_dataset(fn_met)
    bxheight = ds_met.Met_BXHEIGHT.isel(lev=0).values # grid box height (m)
    surf_pres = ds_met.Met_PSC2WET.values * 100.0 # surface pressure (Pa)
    Z0 = ds_met.Met_Z0.values # surface roughness (m)
    CLDFRC = ds_met.Met_CLDFRC.values # column cloud fraction (unitless)
    albedo = ds_met.Met_ALBD.values # visible surface albedo (unitless)
    airden = ds_met.Met_AIRDEN.isel(lev=0).values # air density at surface (kg/m3)
    hflux = ds_met.Met_HFLUX.values # sensible heat flux (W/m2)
    sw_grnd = ds_met.Met_SWGDN.values # incident shortwave at ground (W/m2)
    surf_t = ds_met.Met_TS.values # surface temperature (K)
    ustar = ds_met.Met_USTAR.values # friction velocity (m/s)
    U10M = ds_met.Met_U10M.values # zonal wind at 10 m height (m/s)
    V10M = ds_met.Met_V10M.values # meridional wind at 10 m height (m/s)
    SUNCOSmid = ds_met.Met_SUNCOSmid.values # cosine of solar zenith angle (unitless)

    time_m = ds_met.time # timesteps for month
    time_l = len(time_m)  # number of timesteps in month
    
    days_so_far = datetime.datetime(2015,j+1,1).timetuple().tm_yday - 1 # days of year so far

    # loop over sites
    for i in range(n_sites):
        # find lon and lat indices of sites
        lat_i = np.argmin(np.abs(np.array(lat)-obs_lat[i]))
        lon_i = np.argmin(np.abs(np.array(lon)-obs_lon[i]))
        
        if site_types[i] == 6: # rainforest land type
            F0 = F0_am # select Amazon F0
        else:
            F0 = F0_all #select elsewhere F0
        
        # XLAI has to be scaled based on the areal fraction of land types
        scaling_fac = np.zeros(n_landtypes)
        for LDT in range(n_landtypes):
            # use weird_division function to avoid division by 0.
            scaling_fac[LDT] = weird_division(1., Olson_landtype_v[LDT,lat_i, lon_i])
        
        DV_m = np.zeros(time_l) # initialize hourly deposition velocity array
        
        # loop over hourly timesteps        
        for k in range(time_l): 
            # select XLAI map for the day of hourly timestep
            day_no = time_m[k].dt.day.values.item() - 1  + days_so_far # for python indexing
            XLAI_old = XLAI_d[:,day_no,lat_i, lon_i]
            XLAI_old_scale = XLAI_old * scaling_fac
            
            # make new array for land-type-adjusted XLAI
            XLAI_new = np.zeros(n_landtypes) 
            
            # calculate XLAI for different forest types
            # Deciduous
            sum_area_2 = sum(Olson_landtype_v[IOLSON_2,lat_i, lon_i]) # total area of land type
            if sum_area_2 > 0:
                XLAI_2_all = sum(XLAI_old_scale[IOLSON_2] * \
                                    Olson_landtype_v[IOLSON_2,lat_i, lon_i]) \
                                    / sum_area_2
            # Coniferous
            sum_area_3 = sum(Olson_landtype_v[IOLSON_3,lat_i, lon_i]) # total area of land type
            if sum_area_3 > 0:
                XLAI_3_all = sum(XLAI_old_scale[IOLSON_3] * \
                                    Olson_landtype_v[IOLSON_3,lat_i, lon_i]) \
                                    / sum_area_3
            # Agricultural
            sum_area_4 = sum(Olson_landtype_v[IOLSON_4,lat_i, lon_i]) # total area of land type
            if sum_area_4 > 0:
                XLAI_4_all = sum(XLAI_old_scale[IOLSON_4] * \
                                    Olson_landtype_v[IOLSON_4,lat_i, lon_i]) \
                                    / sum_area_4
            # Rainforest   
            sum_area_6 = sum(Olson_landtype_v[IOLSON_6,lat_i, lon_i]) # total area of land type
            if sum_area_6 > 0:
                XLAI_6_all = sum(XLAI_old_scale[IOLSON_6] * \
                                    Olson_landtype_v[IOLSON_6,lat_i, lon_i]) \
                                    / sum_area_6
            
            # Deciduous            
            if site_types[i] == 2:
                if sum_area_2 != 0:
                    XLAI_new[26] = XLAI_2_all
                elif "Devil's Lake, WI, USA" in location[i]: # exception because no forest for this grid box
                    XLAI_new[26] = XLAI_4_all
                else:
                    print("Problem for site: " + str(i))
            # Coniferous                    
            elif site_types[i] == 3:
                if sum_area_3 != 0:
                    XLAI_new[27] = XLAI_3_all
                elif sum_area_2 != 0: # assume LAI is the same as other forests around
                    XLAI_new[27] = XLAI_2_all
                else:
                    print("Problem for site: " + str(i))
            # Mixed                                        
            elif site_types[i] == [2,3]:
                if (sum_area_2 !=0) and (sum_area_3 !=0): # have LAI of both land types
                    XLAI_new[26] = XLAI_2_all / 2
                    XLAI_new[27] = XLAI_3_all / 2
                elif (sum_area_2 !=0): # have only deciduous forest type in land map
                    XLAI_new[26] = XLAI_2_all / 2
                    XLAI_new[27] = XLAI_2_all / 2
                elif (sum_area_3 !=0): # have only coniferous forest type in land map
                    XLAI_new[26] = XLAI_3_all / 2
                    XLAI_new[27] = XLAI_3_all / 2
                else:
                    print("Problem for site: " + str(i))
            # Rainforest                                                            
            elif site_types[i] == 6:
                if sum_area_6 != 0:
                    XLAI_new[33] = XLAI_6_all
                else:
                    print("Problem for site: " + str(i))
            
            # Calculate supplemental meterological variables
            CZ1, LSNOW, OBK, W10 = METERO(bxheight[k,lat_i,lon_i],\
                                           albedo[k,lat_i,lon_i], \
                                           surf_t[k,lat_i,lon_i], \
                                           ustar[k,lat_i,lon_i], \
                                           airden[k,lat_i,lon_i], \
                                           hflux[k,lat_i,lon_i], \
                                           U10M[k,lat_i,lon_i], \
                                           V10M[k,lat_i,lon_i]) 
                
            # Adjust Z0 value so doesn't fall below 1 m, value for forests
            Z0_forest = max(Z0[k,lat_i,lon_i], 1.0)
            
            # Calculate dry deposition velocity
            DV_m[k] = DEPVEL(DRYCOEFF, IOLSON, IDEP, IRI,\
                             IRLU, IRAC, IRGSS, IRGSO, IRCLS, IRCLO, \
                             IREG_new[i],\
                             ILAND_new[:,i],\
                             IUSE_new[:,i], \
                             surf_t[k,lat_i,lon_i], \
                             XLAI_new, \
                             LSNOW,\
                             sw_grnd[k,lat_i,lon_i],\
                             CLDFRC[k,lat_i,lon_i], \
                             SUNCOSmid[k,lat_i,lon_i],\
                             surf_pres[k,lat_i,lon_i], \
                             ustar[k,lat_i,lon_i],\
                             Z0_forest, \
                             CZ1, OBK, XMW, F0, HSTAR)
        
        # calculate monthly mean of dry deposition velocity for site
        DV_lt[i,j] = np.mean(DV_m) 
                    
# save dry deposition velocity data to .mat file for output
fn_output = 'data/output.mat'
sio.savemat(fn_output, {"DV_lt": DV_lt,"F0": F0, "obs_lat": obs_lat, "obs_lon": obs_lon,"site_types": site_types})
        
