Offline dry deposition model from GEOS-Chem, used in Feinberg et al. (2022): Evaluating atmospheric mercury (Hg) uptake by vegetation in a chemistry-transport model, Environmental Science: Processes & Impacts, https://doi.org/10.1039/D2EM00032F.

This model is a python implementation of the GEOS-Chem dry deposition scheme, based on Wang et al., doi:10.1029/98JD00158 (1998) and Wesely, doi:10.1016/0004-6981(89)90153-4 (1989). To compare with measurement sites, the model's landcover data is adjusted so that 100% of the grid cell is covered with the observed land type. Meteorological and leaf area index input data can be downloaded from http://geoschemdata.wustl.edu. 

File description:

Code

drydep_functions.py - file containing main routines using dry deposition scheme

ex_drydep_simple.py - Script illustrating 0-D example to execute dry deposition routine 

ex_landtype_adj.py - Script illustrating example of adjusting land type to match observation stations, hourly input data


Data

data/Olson_2001_Drydep_Inputs.nc - parameters used in dry deposition scheme for different land cover categories

data/Olson_2001_Land_Type_Masks.2_25.generic.nc - 2x2.5 degree resolution land cover map used in GEOS-Chem, from Gibbs, doi:10.3334/CDIAC/LUE.NDP017.2006 (2006)

data/SI_Forest_Hg_uptake_database.csv - observational dataset of litterfall, throughfall, and open field wet deposition data -> used for location/observed land cover type

[![DOI](https://zenodo.org/badge/451634536.svg)](https://zenodo.org/badge/latestdoi/451634536)

