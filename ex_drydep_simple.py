#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: Jan 2022
Running offline version of dry deposition of GEOS-Chem for simple case
@author: arifeinberg
"""
import xarray as xr
import numpy as np
from drydep_functions import METERO, DEPVEL

#%% Load necessary parameters for running dry deposition code

# Required Olson land variables, dry deposition parameters
fn_ols1 = 'data/Olson_2001_Drydep_Inputs.nc'
ds_ols1 = xr.open_dataset(fn_ols1)

DRYCOEFF = ds_ols1.DRYCOEFF.values # Baldocchi dry deposition polynomial coefficients
IOLSON = ds_ols1.IOLSON.values # Olson land type indices (+1)
IDRYDEP = ds_ols1.IDRYDEP.values # Dry deposition land types

IDEP = ds_ols1.IDEP.values # Mapping index: Olson land type ID to drydep ID
IZO = ds_ols1.IZO.values # Default roughness heights for each Olson land type
IRI = ds_ols1.IRI.values # RI resistance for each dry deposition land type
IRI[2] = 200. # Change RI resistance of coniferous forests to match deciduous, as in current version of GEOS-Chem
IRLU = ds_ols1.IRLU.values # RLU resistance for each dry deposition land type
IRAC = ds_ols1.IRAC.values # RAC resistance for each dry deposition land type
IRGSS = ds_ols1.IRGSS.values # RGSS resistance for each dry deposition land type
IRGSO = ds_ols1.IRGSO.values # RGSO resistance for each dry deposition land type
IRCLS = ds_ols1.IRCLS.values # RCLS resistance for each dry deposition land type
IRCLO = ds_ols1.IRCLO.values # RCLO resistance for each dry deposition land type
IVSMAX = ds_ols1.IVSMAX.values # Max drydep velocity (for aerosol) for each dry deposition land type

#%% Inputs to the dry deposition model

# Meteorological input data
TC0 = 302 # temperature (K)
CFRAC = 0.8 # cloud fraction (unitless)
RADIAT = 606 # incident shortwave at ground (W/m2)
AZO = 1.5 # roughness height (m)
USTAR = 0.15 # friction velocity (m/s)
PRESSU = 1e5 # surface pressure (Pa)
SUNCOS_MID = 0.8 # cosine of solar zenith angle (unitless)
ALBD = 0.1 # albedo (unitless)
BXHEIGHT = 130 # height of lowermost grid box in GEOS-Chem (m)
U10M = 0.1 # zonal wind speed at 10 m height (m/s)
V10M = -0.05 # meridional wind speed at 10 m (m/s)
AIRDEN = 1.15 # dry air density (kg/m3)
HFLUX = 16.8 # sensible heat flux (W/m2)

# Land cover category input data
IREG = 4 # number of land cover categories in this grid box

# Land cover category numbers refer to Olson land types, see: 
# http://wiki.seas.harvard.edu/geos-chem/index.php/Olson_land_map
ILAND = np.zeros(73, dtype=int) # initialize all 73 land categories at 0
ILAND[0] = 0 # Water
ILAND[1] = 29 # Seasonal Tropical Forest
ILAND[2] = 33 # Tropical Rainforest
ILAND[3] = 43 # Savanna (Woods)

# Per mil. land area covered by land cover category
IUSE = np.zeros(73) # initialize all 73 land categories at 0
IUSE[0] = 312. # Water
IUSE[1] = 200. # Seasonal Tropical Forest
IUSE[2] = 175. # Tropical Rainforest
IUSE[3] = 313. # Savanna (Woods)

# Leaf area index for different Olson land cover categories 
XLAI = np.zeros(73) # initialize all 73 land categories at 0
XLAI[0] = 0.1 # Water
XLAI[29] = 0.71 # Seasonal Tropical Forest
XLAI[33] = 0.61 # Tropical Rainforest
XLAI[43] = 0.74 # Savanna (Woods)

# Species specific parameters
XMW = 201e-3 # Hg0 molar mass (kg/mol)
HSTAR = 0.11 # Hg0 Henry's Law Constant (M/atm)

F0 = 3e-5 # Hg0 reactivity

#%% run dep velocity function

# Calculate supplemental meterological variables
CZ1, LSNOW, OBK, W10 = METERO(BXHEIGHT, ALBD, TC0, USTAR, AIRDEN, \
                              HFLUX, U10M, V10M) 

# Calculate dry deposition velocity
DV = DEPVEL(DRYCOEFF, IOLSON, IDEP, IRI, IRLU, IRAC, IRGSS, \
              IRGSO, IRCLS, IRCLO, IREG, ILAND, IUSE, TC0,\
              XLAI, LSNOW, RADIAT, CFRAC, SUNCOS_MID, PRESSU, \
              USTAR, AZO, CZ1, OBK, XMW, F0, HSTAR)

print("Calculated dry deposition velocity (cm/s): ")
print(DV)