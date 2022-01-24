#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: Jan 2022

Dry deposition functions

@author: arifeinberg
"""
import xarray as xr
import numpy as np

def METERO(BXHEIGHT, ALBD, TS, USTAR, AIRDEN, HFLUX, U10M, V10M):
    """Calculate meteorological variables needed for dry deposition scheme
    
    Parameters
    ----------
    BXHEIGHT : float
         grid box height
    ALBD : float
         surface albedo 
    TS : float
         surface temperature 
    USTAR : float
         friction velocity 
    AIRDEN : float
         air density 
    HFLUX : float
         sensible heat flux 
    U10M : float
         zonal wind speed at 10 m 
    V10M : float
         meridional wind speed at 10 m  
         
    """
    CZ1 = BXHEIGHT / 2.0 # midpoint height of first model level
    LSNOW = ALBD > 0.4 # logical if snow and ice

    OBK = GET_OBK(TS, USTAR, AIRDEN, HFLUX) # Monin-Obhukov length (m)

    # calculate 10 m windspeed
    W10 = np.sqrt(U10M**2 + V10M**2)
    
    return CZ1, LSNOW, OBK, W10



def GET_OBK(TS, USTAR, AIRDEN, HFLUX):
    """Calculate Monin-Obhukov length
    
    Parameters
    ----------
    TS : float
         surface temperature
    USTAR : float
         friction velocity 
    AIRDEN : float
         air density 
    HFLUX : float
         sensible heat flux 
         
    """
    # constants
    KAPPA = 0.4 # Von Karman's constant
    CP = 1000.0 # specific heat of air at constant P (J/kg/K)
    g0 = 9.80665 # acceleration due to gravity at Earth's surface (m/s^2)

    NUM = -AIRDEN * CP * TS * USTAR**3 # Numerator
    if HFLUX == 0.: # make sure denominator not zero
        HFLUX = 1e-20 # set to small number
        
    DEN = KAPPA * g0 * HFLUX # Denominator

    OBK = NUM / DEN # Monin-Obhukov length in m
    
    return OBK

def Compute_Olson_landmap(Olson_landtype):
    """compute Olson land map variables
    
    Parameters
    ----------
    Olson_landtype : xarray
         fractional area of each Olson landtype         
    """

    ### Compute Olson_landmap # 
    Olson_landtype_v = Olson_landtype.squeeze().values # select values
    
    # constants
    ltype = Olson_landtype["variable"]
    lat = Olson_landtype.lat
    lon = Olson_landtype.lon
    
    NSURFTYPE = len(ltype) # number of surface types
    lat_l = len(lat) # number of lat values
    lon_l = len(lon) # number of lon values

    # initialize variables
    FRCLND = np.ones((lat_l,lon_l)) # fraction of land in grid box (start as 1)
    IREG = np.zeros((lat_l,lon_l), dtype=int) # number of types in grid box
    ILAND = np.full([NSURFTYPE, lat_l,lon_l], np.nan, dtype=int) # index of types actually in grid box
    IUSE = np.zeros((NSURFTYPE, lat_l,lon_l)) # fractional coverage of types actually in grid box in per mil
    
    # calculate over the horizontal grid
    for i in range(lat_l):
        for j in range(lon_l):
            typeCounter = 0 # count the number of types in the grid box
            # Loop over landmap types to calculate IREG, ILAND, and IUSE
            for T in range(NSURFTYPE):
                # If this type has non-zero coverage in this grid box, update vars
                if Olson_landtype_v[T,i,j] > 0:
                    # Store type index in ILAND array
                    ILAND[typeCounter,i,j] = T
                    # Store fractional coverage in IUSE array (in per mil)
                    IUSE[typeCounter,i,j] = Olson_landtype_v[T,i,j] * 1000
                    # Increment number of types in this cell
                    typeCounter +=1
                    
            # Set IREG to the number of types
            IREG[i,j] = typeCounter
            
            # Force IUSE to sum to 1000 by updating max value if necessary
            maxFracInd = np.argmax(IUSE[:,i,j]) # index of max land cover area
            
            # Check if sum of IUSE is less than 1000
            sumIUSE = IUSE[:,i,j].sum()
            if sumIUSE !=1000: # Update max IUSE to fix imbalance
                IUSE[maxFracInd,i,j] = IUSE[maxFracInd,i,j] + (1000 - sumIUSE)
            
            
    # Subtract water coverage from fraction of land
    FRCLND = FRCLND - Olson_landtype_v[0, :, :]
    # create xarrays
    FRCLND_xr = xr.DataArray(FRCLND, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    IREG_xr = xr.DataArray(IREG, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    ILAND_xr = xr.DataArray(ILAND, dims=("variable", "lat", "lon"), coords={"variable": ltype, "lat": lat, "lon": lon})
    IUSE_xr = xr.DataArray(IUSE, dims=("variable", "lat", "lon"), coords={"variable": ltype, "lat": lat, "lon": lon})
    
    return FRCLND_xr, IREG_xr, ILAND_xr, IUSE_xr

def BIOFIT(COEFF1, XLAI1, SUNCOS1, CFRAC1, NPOLY):
    """compute the light correction used in the dry deposition modules
    
    Parameters
    ----------
    COEFF1 : ndarray
         Baldocchi drydep coefficients      
    XLAI1 : ndarray
         Leaf area index (cm2/cm2)         
    SUNCOS1 : ndarray
         Cosine (Solar Zenith Angle)         
    CFRAC1 : ndarray
         Cloud fraction (unitless)         
    NPOLY : integer
         # of drydep coefficients         
         
    """
    # Code comes from Wang et al., JGR, 1998
    # Parameters
    KK = 4
    
    REALTERM = np.zeros(NPOLY)
    TERM = np.zeros(KK)
    TERM[0] = 1.
    TERM[1] = XLAI1
    TERM[2] = SUNCOS1
    TERM[3] = CFRAC1
    # Call additional function to perform light correction
    TERM[1:] = SUNPARAM(TERM[1:])
    K=0
    for K3 in range(0,KK):
       for K2 in range(K3, KK):
          for K1 in range(K2, KK):
             REALTERM[K] = TERM[K1] * TERM[K2] * TERM[K3]
             K = K+1
             
    BIO_FIT = 0. # resultant light correction
    for K in range(NPOLY):
       BIO_FIT = BIO_FIT + COEFF1[K] * REALTERM[K]
       
    if ( BIO_FIT < 0.1 ):
        BIO_FIT=0.1
    
    return BIO_FIT 

def SUNPARAM(X):
    """ perform the light correction, called by BIOFIT
    
    Parameters
    ----------
    X : ndarray
         Term[1:3] in BIOFIT   
    """
    NN = 3 #  number of variables (LAI, SUNCOS, CLDFRC)
    # the sequence is lai,suncos,cloud fraction

    ND = [55., 20., 11.] # scaling factor for each variable

    X0 = [11., 1., 1.] # maximum for each variable

    for I in range(NN):
       X[I] = min(X[I], X0[I])
       # XLOW = minimum for each variable
       if I != 2:
          XLOW = X0[I] / ND[I]
       else:
          XLOW= 0.
          
       X[I] = max(X[I],XLOW)
       X[I] = X[I] / X0[I]

    return X

def DIFFG(TK, PRESS, XM):
    """ Calculate the molecular diffusivity in air for a gas X in m2/s
    
    Parameters
    ----------
    TK : float
         Temperature in K
    PRESS : float
         Pressure in Pa
    XM : float
         Molecular weight of gas in kg/mol
         
    """
    
    # Constants
    XMAIR = 28.8e-3 # moist air molec wt?
    RADAIR = 1.2e-10 # raidus of air molecule
    RADX = 1.5e-10 # radius of X, assuming molecule is not so big
    RSTARG = 8.3144598 # molar gas constant (J/K/mol)
    AVO = 6.022140857e23 # Avogadro's number (molec/mol)
    PI = 3.14159265358979323 # pi
    
    # air  [molec/m^3]
    AIRDEN = ( PRESS * AVO ) / ( RSTARG * TK )

    # DIAM is the collision diameter for gas X with air.
    DIAM   = RADX + RADAIR

    # Calculate the mean free path for gas X in air:
    # eq. 8.5 of Seinfeld [1986];
    Z      = XM  / XMAIR
    FRPATH = 1. /( PI * np.sqrt( 1. + Z ) * AIRDEN * ( DIAM**2 ) )

    # Calculate average speed of gas X; eq. 15.47 of Levine [1988]
    SPEED  = np.sqrt( 8. * RSTARG * TK / ( PI * XM ) )

    # Calculate diffusion coefficient of gas X in air;
    # eq. 8.9 of Seinfeld [1986]
    DIFF_G = ( 3.0 * PI / 32.0 ) * ( 1.0 + Z ) * FRPATH * SPEED
    
    return DIFF_G

def DEPVEL(DRYCOEFF, IOLSON, IDEP, IRI, IRLU, IRAC, IRGSS, IRGSO, IRCLS, IRCLO, \
           IREG, ILAND, IUSE, \
           TS, XLAI, LSNOW, RADIAT, CLDFRC, SUNCOSMID, PRESSU, USTAR, Z0, \
           CZ1, OBK, XMW, F0, HSTAR):
    """ Calculate the deposition velocity for Hg0 - 1D version for grid cell
    
    Parameters
    ----------
    DRYCOEFF : ndarray
         Baldocchi drydep coefficients
    IOLSON : ndarray
         Olson land type indices
    IDEP : ndarray
         Mapping index: Olson land type ID to drydep ID
    IRI : ndarray
         internal stomatal resistance for each dry deposition land type
    IRLU : ndarray
         cuticular resistance for each dry deposition land type
    IRAC : ndarray
         transfer that depends on canopy height
    IRGSS : ndarray
         transfer to soil, leaf litter for SO2
    IRGSO : ndarray
         transfer to soil, leaf litter for O3
    IRCLS : ndarray
         transfer to leaves, twigs in low canopy for SO2
    IRCLO : ndarray
         transfer to leaves, twigs in low canopy for O3

    IREG : float
         number of types in grid box
    ILAND : ndarray
         index of types actually in grid box
    IUSE : ndarray
         fractional coverage of types actually in grid box in per mil

    TS : float
         surface temperature in K
    XLAI : ndarray
         LAI differentiated by land type
    LSNOW : float
         logical if covered by snow and ice
    RADIAT : float
         incident shortwave at ground
    CLDFRC : float
         column cloud fraction
    SUNCOSMID : float
         cosine of solar zenith angle, at midpoint of chemistry timestep
    PRESSU : float
         surface pressure in Pa
    USTAR : float
         friction velocity
    Z0 : float
         surface roughness in m
    CZ1 : float
         midpoint height of first model level
    OBK : float
         Monin-Obhukov length in m
         
    XMW : float
         molecular mass of species in kg/mol
    F0 : float
         biological reactivity of species
    HSTAR : float
         Henry's law constant at pH 7 (for surface)
    """

    # constants
    NN = len(DRYCOEFF)
    dims_Olson = ILAND.shape
    NSURFTYPE = dims_Olson[0] # number of surface types
    H2OMW = 18.016 # Molecular weight of water (g/mol)
    VON_KARMAN = 0.4 # Von Karman's constant (unitless)
    SMALL = 1e-10 # small number to avoid instabilities
    
    
    
    # Start calculations
    VD = 0
    TEMPK = TS # Temperature in Kelvin
    TEMPC = TS - 273.15 # Temperature in Celsius

    # Calculate the kinematic viscosity XNU (m2 s-1) of air:
    C1 = TEMPK/273.15
    XNU = 0.151*C1**1.77*1e-4
    
    # Compute bulk surface resistance for gases
    RT = 1000.0 * np.exp(-TEMPC-4.)
    
    # Initialize variables
    RSURFC = np.zeros(NSURFTYPE)
    RI = np.zeros(NSURFTYPE) # internal resistance
    RLU = np.zeros(NSURFTYPE) # cuticular resistance
    RAC = np.zeros(NSURFTYPE) # transfer that depends on canopy height
    RGSS = np.zeros(NSURFTYPE) # transfer to soil, leaf litter for SO2
    RGSO = np.zeros(NSURFTYPE) # transfer to soil, leaf litter for O3
    RCLS = np.zeros(NSURFTYPE) # transfer to leaves, twigs in low canopy for SO2
    RCLO = np.zeros(NSURFTYPE) # transfer to leaves, twigs in low canopy for O3

    # Get surface resistances - loop over land types LDT
    for LDT in range(IREG):
        
        IOLSON = ILAND[LDT] # Olson land type (0 - 72)
        II = IDEP[IOLSON] # dry deposition land type (1 - 11)

        # Scale XLAI by the fraction of area of land type
        XLAI_scale = XLAI[IOLSON] / IUSE[LDT] * 1000.
        
        # If the surface is snow or ice, set II to 1
        if LSNOW:
            II = 1
        
        # Read internal resistance RI from IRI array
        RI[LDT] = IRI[II-1]
        # If the value is above 9999, means no deposition to stomata
        # so impose a large value
        if RI[LDT] >= 9999.:
            RI[LDT] = 1e12
            
        # Read cuticular resistances from IRLU. Since they are per unit 
        # area of leaf, divide them by the leaf area index to get a 
        # cuticular resistance for the bulk canopy
        if (IRLU[II-1] >=9999.) or (XLAI_scale <= 0.):
            RLU[LDT] = 1e6
        else:
            RLU[LDT] = IRLU[II-1] / XLAI_scale
            # additional resistance at low temperatures, limit increase
            # to a factor of 2
            RLU[LDT] = min(RLU[LDT] + RT, 2 * RLU[LDT])
            
        # Remaining resistances for the Wesely model:
        RAC[LDT] = max(IRAC[II-1], 1.0)
        if RAC[LDT] >= 9999.:
            RAC[LDT] = 1e12
        
        RGSS[LDT] = max(IRGSS[II-1],1.0)
        # Additional resistance at low temperatures:
        RGSS[LDT] = min(RGSS[LDT] + RT, 2 * RGSS[LDT])
        if RGSS[LDT] >= 9999.:
            RGSS[LDT] = 1e12
        
        RGSO[LDT] = max(IRGSO[II-1],1.0)
        # Additional resistance at low temperatures:
        RGSO[LDT] = min(RGSO[LDT] + RT, 2 * RGSO[LDT])
        if RGSO[LDT] >= 9999.:
            RGSO[LDT] = 1e12
        
        # Additional resistance at low temperatures:
        RCLS[LDT] = min(IRCLS[II-1] + RT, 2 * IRCLS[II-1])
        if RCLS[LDT] >= 9999.:
            RCLS[LDT] = 1e12

        # Additional resistance at low temperatures:
        RCLO[LDT] = min(IRCLO[II-1] + RT, 2 * IRCLO[II-1])
        if RCLO[LDT] >= 9999.:
            RCLO[LDT] = 1e12
        
        # Adjust stomatal resistances for insolation and temperature:
        RA = RADIAT
        
        RIX = RI[LDT]
        if RIX < 9999.0:
            GFACT = 100.
            if (TEMPC > 0.) and (TEMPC < 40.):
                GFACT = 400. / TEMPC / (40. - TEMPC)
            
            GFACI = 100.
            if (RA > 0.) and (XLAI_scale > 0. ):
                GFACI = 1. / BIOFIT( DRYCOEFF,  XLAI_scale, \
                            SUNCOSMID, CLDFRC, NN )

      
            RIX = RIX * GFACT * GFACI
            
        # Compute the aerodynamic resistance to lower elements in the 
        # lower part of the canopy or structure, assuming level terrain
        
        RDC = 100. * (1. + 1000.0 / (RA + 10.0))
        
        # Calculate species dependent dry deposition parameters
        
        XMWH2O = H2OMW * 1.e-3
        RIXX = RIX * DIFFG(TEMPK, PRESSU, XMWH2O) / \
            DIFFG(TEMPK, PRESSU, XMW) + \
            1 / (HSTAR / 3000. + 100. * F0)
        
        # cuticular resistance
        RLUXX = 1.e12
        if (RLU[LDT] < 9999.):
             RLUXX = RLU[LDT] / (HSTAR / 1e5 + F0)
        
        # soil resistance
        RGSX = 1. / (HSTAR / 1e5 / RGSS[LDT] + F0 / RGSO[LDT])
        
        # lower canopy resistance
        RCLX = 1. / (HSTAR / 1e5 / RCLS[LDT] + F0 / RCLO[LDT])

        # Get the bulk surface resistance of the canopy, RSURFC, from
        # the network of resistances in parallel and in series (Fig. 1
        # of Wesely [1989])
        DTMP1 = 1. / RIXX
        DTMP2 = 1. / RLUXX
        DTMP3 = 1. / (RAC[LDT] + RGSX)
        DTMP4 = 1. / (RDC + RCLX)
        RSURFC[LDT] = 1. / (DTMP1 + DTMP2 + DTMP3 + DTMP4)
        
        
        # Set maximum value of surface resistance
        RSURFC[LDT] = max(1.0, min(RSURFC[LDT],9999.))
        # Calculate the aerodynamic resistances Ra and Rb
        
        CKUSTR = VON_KARMAN * USTAR
        
        REYNO = USTAR * Z0 / XNU
        
        CORR1 = CZ1 / OBK
        
        #Define Z0OBK
        Z0OBK = Z0 / OBK
         
        # Compute resistance depending whether have aerodynamically 
        # rough or smooth surface
        if REYNO >= 0.1:
            # aerodynamically rough surface            
            if (CORR1<0.0):
                # unstable conditions; 
                DUMMY1 = (1. - 15. * CORR1) ** 0.5
                DUMMY2 = (1. - 15. * Z0OBK) ** 0.5
                DUMMY3 = abs((DUMMY1 - 1.) / (DUMMY1 + 1.))
                DUMMY4 = abs((DUMMY2 - 1.) / (DUMMY2 + 1.))
                RA = 1. * (1./CKUSTR) * np.log(DUMMY3/DUMMY4)

            elif((CORR1 >= 0.0) and (CORR1<=1.0)):
                RA = (1. / CKUSTR) * (1. * np.log(CORR1 / Z0OBK) + \
                                      5. * (CORR1 - Z0OBK))            
            else: # CORR1 > 1.0
                RA = (1. / CKUSTR) * (5. * np.log(CORR1 / Z0OBK) + \
                                      1. * (CORR1 - Z0OBK))
                    
            # set maximum values for RA
            RA = min(RA, 1e4)
            
            # make sure RA is not negative
            if RA < 0.:
                RA = 0.
                
            # get total resistance for deposition
            DAIR = 0.2 * 1e-4 # thermal diffusivity of air
            RB = (2. / CKUSTR) * \
                (DAIR / DIFFG(TEMPK, PRESSU, XMW)) ** 0.667
            C1X = RA + RB + RSURFC[LDT]
        else: # aerodynamically smooth surface
            RA = 1e4
            
            C1X = RA + RSURFC[LDT]
        # sum contribution of surface type LDT to the deposition velocity    
        VK = VD 
        #IUSE_round = int(IUSE[LDT])
        #if LDT==0:
        #    IUSE_round = IUSE_round + 1
        VD = VK + 0.001 * IUSE[LDT]/C1X
        
    # save the total deposiiton velocity to an array
    DVEL = VD * 100. # units of cm/s
    return DVEL