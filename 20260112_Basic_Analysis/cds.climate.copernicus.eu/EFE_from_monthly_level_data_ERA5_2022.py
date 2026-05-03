#!/usr/bin/env python
# coding: utf-8

# # TRYING ... to compute EFE from ERA5 monthly levelwise data
# Doing the vertical integral of v*MSE first and then computed its divergent component

# In[1]:


def mass_weighted_integral_simple(data, g=9.81):
    """
    Simple mass-weighted integral using composite Simpson's rule.

    Parameters
    ----------
    data : xr.DataArray
        Data to integrate (time, plev, lat, lon)
    g : float
        Gravity (m/s²)

    Returns
    -------
    xr.DataArray
        Vertically integrated data
    """
    from scipy.integrate import simpson

    # Get pressure levels
    plev = data.pressure_level.values*100

    # Fill NaNs
    data_filled = data.fillna(0.0)

    # Load data if it's a dask array (to avoid dask issues)
    if hasattr(data_filled.data, 'compute'):
        data_filled = data_filled.compute()

    # Use scipy.integrate.simpson
    # Find which axis is 'plev'
    plev_axis = data.dims.index('pressure_level')

    # Integrate using Simpson's rule
    integrated = simpson(data_filled.values, x=plev, axis=plev_axis)

    # Create output DataArray
    output_dims = [d for d in data.dims if d != 'pressure_level']
    output_coords = {d: data[d] for d in output_dims}

    result = xr.DataArray(
        integrated / g,
        coords=output_coords,
        dims=output_dims
    )

    return result


# In[2]:


import numpy as np
import xarray as xr
# our local module:

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


import xarray as xr
from pathlib import Path
# import myfunctions as mf


# In[4]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# from xarrayutils import divergence_spherical, helmholtz_decomposition_spectral  # placeholder functions


# In[5]:


ds = xr.open_dataset("ERA5_Temp_SpHum_Geopot_u_v_2022.nc")

ds


# In[ ]:





# In[6]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from windspharm.standard import VectorWind
import cftime
# import mse_divergence


# ----------------------------
# CONSTANTS
# ----------------------------
Cp = 1004.0        # J kg-1 K-1
Lv = 2.5e6         # J kg-1
g  = 9.81          # m s-2
R  = 6.371e6       # Earth radius (m)

# ----------------------------
# BLOCK 1: READING VARIABLES (lat,lon,lev,time)
# ----------------------------

start_year = 2071
end_year   = 2101 ;# These are not used at the moment since I am testing with historical run

ta_mean = ds.t
q_mean  = ds.q
z_mean  = ds.z
u_mean  = ds.u
v_mean  = ds.v


# ----------------------------
# BLOCK 2.1 : MOIST STATIC ENERGY (lat,lon,lev,time)
# ----------------------------
h = Cp * ta_mean + Lv * q_mean + z_mean
h_on_v = h.interp(latitude=v_mean.latitude, longitude=v_mean.longitude, pressure_level=v_mean.pressure_level, method="linear")


# ----------------------------
# BLOCK 2.2 : BAROTROPIC CORRECTION (lat,lon,lev,time)
# ----------------------------
# Fill NaNs
u_filled = u_mean.fillna(0.0)
v_filled = v_mean.fillna(0.0)

# Barotropic mass correction: Not done

# ----------------------------
# BLOCK 3 : COMPUTE UH and VH (lat,lon,lev,time)
# ----------------------------
# Zonal wind * MSE
uh = u_filled * h_on_v

# Meridional wind * MSE
vh = v_filled * h_on_v


# Ensure monotonic plev and fill NaNs
uh_sorted = uh.sortby("pressure_level").fillna(0.0)
vh_sorted = vh.sortby("pressure_level").fillna(0.0)

plev_vals = np.array(vh_sorted.pressure_level.values, dtype=float)
order = np.argsort(plev_vals)

uh_clean = uh_sorted.isel(pressure_level=order).assign_coords(pressure_level=plev_vals[order])
vh_clean = vh_sorted.isel(pressure_level=order).assign_coords(pressure_level=plev_vals[order])


##############################################
########### Vertical integration #############
##############################################

# uh_int = uh_clean.integrate("plev") / g
# vh_int = vh_clean.integrate("plev") / g

##############################################
########### Mid-point approach ###############
##############################################
# Get the pressure levelsL
plev = vh_clean.pressure_level*100 ;#Converting to pascal

# --------------------------------------------
# Compute layer thickness dp for each level (same as barotropic correction)
# --------------------------------------------
n_lev = len(plev)
dp_vals = np.zeros(n_lev)

# Top level (index 0): use distance to next level
dp_vals[0] = abs(plev.values[1] - plev.values[0])

# Interior levels: use centered difference
for i in range(1, n_lev - 1):
    dp_vals[i] = abs(plev.values[i+1] - plev.values[i-1]) / 2.0

# Bottom level: use distance to previous level
dp_vals[-1] = abs(plev.values[-1] - plev.values[-2])

# Create DataArray
dp = xr.DataArray(dp_vals, coords={"plev": plev}, dims=["pressure_level"])
dp = dp.assign_coords(plev=plev)



# BEFORE computing uh_int and vh_int, reorder latitude
if uh_clean.latitude.values[0] < uh_clean.latitude.values[-1]:
    uh_clean = uh_clean.sortby('lat', ascending=False)
    vh_clean = vh_clean.sortby('lat', ascending=False)
    print("Latitudes reordered to descending")

# Then compute vertical integration
# uh_int = (uh_clean * dp).sum("plev") / g
# vh_int = (vh_clean * dp).sum("plev") / g
# Correct integration (no /g here)
# uh_int = (uh_clean * dp).sum("plev")
# vh_int = (vh_clean * dp).sum("plev")

# Vertical integration using this function
uh_int = mass_weighted_integral_simple(uh_clean, g=9.81)
vh_int = mass_weighted_integral_simple(vh_clean, g=9.81)

print("uh_int range:", uh_int.min().values, "to", uh_int.max().values)
print("vh_int range:", vh_int.min().values, "to", vh_int.max().values)

# Make sure they stay in descending order
print("uh_int lat order:", uh_int.latitude.values[:3], "...", uh_int.latitude.values[-3:])





# In[7]:


# Regrid using xarray's interp
# uh_int_regrid = uh_int.interp(lat=target_lat, lon=target_lon, method='linear')


# In[8]:


# vh_int_regrid = vh_int.interp(lat=target_lat, lon=target_lon, method='linear')


# In[9]:


from windspharm.xarray import VectorWind

def windspharm_time_loop(uh_int, vh_int):
    """
    Compute divergent component for each time step
    """
    Flambda_div_list = []
    Fphi_div_list = []

    for t in range(len(uh_int.valid_time)):
        # Extract single time step
        u_t = uh_int.isel(valid_time=t)
        v_t = vh_int.isel(valid_time=t)

        # VectorWind requires (lat, lon) with no NaNs
        u_filled = u_t.fillna(0.0)
        v_filled = v_t.fillna(0.0)

        # Compute divergent component
        vw = VectorWind(u_filled, v_filled)
        u_div, v_div = vw.irrotationalcomponent()  # Changed here!

        Flambda_div_list.append(u_div)
        Fphi_div_list.append(v_div)

    # Concatenate back along time dimension
    Flambda_div = xr.concat(Flambda_div_list, dim='valid_time')
    Fphi_div = xr.concat(Fphi_div_list, dim='valid_time')

    # Assign time coordinates
    Flambda_div = Flambda_div.assign_coords(time=uh_int.valid_time)
    Fphi_div = Fphi_div.assign_coords(time=vh_int.valid_time)

    return Flambda_div, Fphi_div



# In[10]:


uh_int


# In[ ]:


# # # CDS data has ASCENDING latitude (-90 to 90), but VectorWind needs DESCENDING (90 to -90)
# # # So reverse the latitude
# # uh_int_regrid = uh_int_regrid.sortby('lat', ascending=False)
# # vh_int_regrid = vh_int_regrid.sortby('lat', ascending=False)

# print("\nAfter regridding:")
# print("  uh_int_regrid lat:", uh_int_regrid.lat.values[:5], "...", uh_int_regrid.lat.values[-5:])
# print("  Shape:", uh_int_regrid.shape)

# Now compute divergent component on the CDS grid
Flambda_div, Fphi_div = windspharm_time_loop(uh_int, vh_int)

print("\nFphi_div range:", Fphi_div.min().values, "to", Fphi_div.max().values)




# # Make sure latitude is descending for VectorWind
# if uh_int_regrid.lat.values[0] < uh_int_regrid.lat.values[-1]:
#     uh_int_regrid = uh_int_regrid.sortby('lat', ascending=False)
#     vh_int_regrid = vh_int_regrid.sortby('lat', ascending=False)

# # Now compute divergent component on the regular grid
# Flambda_div, Fphi_div = windspharm_time_loop(uh_int_regrid.fillna(0.0), vh_int_regrid.fillna(0.0))

# print("After regridding:")
# print("Fphi_div range:", Fphi_div.min().values, "to", Fphi_div.max().values)


# # # Then compute divergent component
# # Flambda_div, Fphi_div = windspharm_time_loop(uh_int, vh_int)



# # ----------------------------
# # BLOCK 4 : COMPUTE INTEGRAL (lat,lon,time)
# # ----------------------------
# # Computing (1/g) ∫ v dp
# # uh_int = (uh_clean * dp / g).sum("plev")
# # vh_int = (vh_clean * dp / g).sum("plev")
# # CORRECT version
# uh_int = (uh_clean * dp).sum("plev") / g
# vh_int = (vh_clean * dp).sum("plev") / g

# # ----------------------------
# # BLOCK 5 : GET h_DIV (lat,lon,time)
# # ----------------------------
# Flambda_div, Fphi_div = windspharm_time_loop(uh_int, vh_int)

Fphi_div.name = "Fphi_div"
Fphi_div.attrs["long_name"] = "divergent meridional MSE flux"
Fphi_div.attrs["units"] = "W m^-1"


# In[ ]:


v_mean.attrs["long_name"] = "v original"
h_on_v.attrs["long_name"] = "h after regriding to grid of v"
vh.attrs["long_name"] = "v_corr * h_on_v"
vh_int.attrs["long_name"] = "vh vertically integrated"
uh_int.attrs["long_name"] = "uh vertically integrated"


# In[ ]:


import xarray as xr

ds_save = xr.Dataset(
    {
        "Flambda_div": Flambda_div,
        "Fphi_div": Fphi_div,
        "h_on_v": h_on_v,
        "vh_int": vh_int,
        "uh_int": uh_int,
    }
)

#Add compression (important for daily data)
encoding = {
    var: {"zlib": True, "complevel": 4}
    for var in ds_save.data_vars
}

#Save to Netcdf
ds_save.to_netcdf(
    "EFE_ERA5_2022.nc",
    format="NETCDF4",
    encoding=encoding
)

