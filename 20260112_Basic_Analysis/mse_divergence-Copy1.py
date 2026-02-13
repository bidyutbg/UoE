import numpy as np
import xarray as xr
from windspharm.standard import VectorWind
import myfunctions as mf

def compute_mse_flux_divergence(
    ta, q, zg, ua, va,
    lon_slice=slice(0, 360),
    season=(1, 12),
    Cp=1004.0,
    Lv=2.5e6,
    g=9.81,
):
    """
    Compute vertically integrated divergent MSE flux (F_div).

    Parameters
    ----------
    ta, q, zg, ua, va : xarray.DataArray
        Temperature (K), specific humidity (kg/kg), geopotential height (m),
        zonal wind (m/s), meridional wind (m/s)
    lon_slice : slice
        Longitude range
    season : tuple
        (start_month, end_month)
    Cp, Lv, g : float
        Physical constants

    Returns
    -------
    F_div : xarray.DataArray
        Vertically integrated divergent MSE flux (W m⁻¹)
        dims: (lat, lon)
    """

    # ----------------------------
    # TIME MEAN
    # ----------------------------
    ta_mean = mf.seasonal_mean_by_year(ta, *season).mean("year")
    q_mean  = mf.seasonal_mean_by_year(q,  *season).mean("year")
    z_mean  = mf.seasonal_mean_by_year(zg, *season).mean("year")
    u_mean  = mf.seasonal_mean_by_year(ua, *season).mean("year")
    v_mean  = mf.seasonal_mean_by_year(va, *season).mean("year")

    # ----------------------------
    # FILL NaNs
    # ----------------------------
    u_mean = u_mean.fillna(0.0)
    v_mean = v_mean.fillna(0.0)

    # ----------------------------
    # BAROTROPIC MASS CORRECTION
    # ----------------------------
    u_corr, v_corr = mf.barotropic_mass_correction(u_mean, v_mean)

    vw = VectorWind(u_corr, v_corr)
    _, v_div = vw.divergentcomponent()

    v_div = xr.DataArray(
        v_div,
        coords=v_mean.coords,
        dims=v_mean.dims,
        name="v_div"
    )

    # ----------------------------
    # SELECT LONGITUDE SECTOR
    # ----------------------------
    ta_sec = ta_mean.sel(lon=lon_slice)
    q_sec  = q_mean.sel(lon=lon_slice)
    z_sec  = z_mean.sel(lon=lon_slice)
    v_sec  = v_div.sel(lon=lon_slice)

    # ----------------------------
    # MOIST STATIC ENERGY
    # ----------------------------
    h = Cp * ta_sec + Lv * q_sec + g * z_sec

    # ----------------------------
    # INTERPOLATE MSE TO WIND GRID
    # ----------------------------
    h_on_v = h.interp(
        lat=v_sec.lat,
        lon=v_sec.lon,
        plev=v_sec.plev,
        method="linear"
    )

    # ----------------------------
    # MERIDIONAL MSE FLUX
    # ----------------------------
    vh = v_sec * h_on_v

    # ----------------------------
    # VERTICAL INTEGRATION
    # ----------------------------
    vh = vh.sortby("plev").compute().fillna(0.0)

    # enforce strictly monotonic pressure
    plev_vals = np.array(vh.plev.values, dtype=float)
    order = np.argsort(plev_vals)

    vh = vh.isel(plev=order)
    vh = vh.assign_coords(plev=plev_vals[order])

    vh_int = vh.integrate("plev")

    F_div = vh_int / g
    F_div.name = "F_div"
    F_div.attrs["units"] = "W m-1"

    return F_div

import xarray as xr
import cftime
import numpy as np
from pathlib import Path
from windspharm.standard import VectorWind
import myfunctions as mf
import itcz

# ----------------------------
# Helper: convert integer years to cftime slice
# ----------------------------
def year_slice_to_cftime(da, start_year, end_year):
    """Convert integer years to cftime slice for a DataArray"""
    start = cftime.Datetime360Day(start_year, 1, 1)
    end   = cftime.Datetime360Day(end_year, 12, 30)
    return slice(start, end)

# ----------------------------
# Core high-level wrapper
# ----------------------------
def compute_Fdiv_for_model_experiments(
    model_name,
    model_meta,
    experiments,
    base_dir,
    lon_slice=slice(0,360),
    season=(1,12),
    time_slice=None,   # slice(start_year, end_year)
):
    """
    Compute vertically-integrated divergent meridional MSE flux (F_div)
    for all requested experiments of a given model.

    Parameters
    ----------
    model_name : str
        Model name (e.g., "UKESM1-0-LL")
    model_meta : dict
        Metadata for the model (institution, ensemble, grid)
    experiments : dict
        Experiment metadata dict
    base_dir : Path
        Base CEDA path
    lon_slice : slice
        Longitude range to select
    season : tuple(int,int)
        Start and end month for seasonal mean
    time_slice : slice
        Start and end years, e.g., slice(2071, 2101)

    Returns
    -------
    Fdiv_by_exp : dict
        Keys are experiment names, values are xarray.DataArray (lat, lon)
    """

    Fdiv_by_exp = {}

    # List of variables we need
    varnames = ["ta","hus","zg","ua","va"]

    for exp, meta in experiments.items():

        # # Only compute for scenario experiments
        # if exp not in ["SSP245","G6solar","G6sulfur"]:
        #     continue

        # Ensemble overrides
        if model_name == "CESM2-WACCM":
            if meta["scenario"] == "G6sulfur":
                ensemble = "r1i1p1f2"
            else:
                ensemble = "r1i1p1f1"
        else:
            ensemble = model_meta["ensemble"]

        # ----------------------------
        # Function to open & slice a variable safely
        # ----------------------------
        def open_var(varname):
            base = (
                base_dir
                / meta["project"]
                / model_meta["institution"]
                / model_name
                / meta["scenario"]
                / ensemble
                / "Amon"
                / varname
                / model_meta["grid"]
                / "latest"
            )

            files = sorted(base.glob("*.nc"))
            if len(files) == 0:
                raise FileNotFoundError(f"No NetCDF files found in {base}")

            # Open with safe combine
            ds = xr.open_mfdataset(
                [str(f) for f in files],
                combine="nested",
                concat_dim="time",
                decode_times=True,
                use_cftime=True,
                parallel=True,
            )

            # Read variable
            da = mf.read_var(ds, varname)
            if da is None:
                raise ValueError(f"Variable {varname} not found in {base}")

            # Apply time slice if requested
            if time_slice is not None:
                start_year, end_year = time_slice.start, time_slice.stop-1
                da = da.sel(time=year_slice_to_cftime(da, start_year, end_year))

            return da

        # ----------------------------
        # Open all variables
        # ----------------------------
        ta  = open_var("ta")
        q   = open_var("hus")
        zg  = open_var("zg")
        ua  = open_var("ua")
        va  = open_var("va")

        # ----------------------------
        # Compute seasonal mean
        # ----------------------------
        start_month, end_month = season
        ta_mean = mf.seasonal_mean_by_year(ta, start_month, end_month).mean("year")
        q_mean  = mf.seasonal_mean_by_year(q,  start_month, end_month).mean("year")
        z_mean  = mf.seasonal_mean_by_year(zg, start_month, end_month).mean("year")
        ua_mean = mf.seasonal_mean_by_year(ua, start_month, end_month).mean("year")
        va_mean = mf.seasonal_mean_by_year(va, start_month, end_month).mean("year")

        # ----------------------------
        # Compute divergent winds
        # ----------------------------
        # Fill NaNs
        u_filled = ua_mean.fillna(0.0)
        v_filled = va_mean.fillna(0.0)

        # Barotropic mass correction
        u_corr, v_corr = mf.barotropic_mass_correction(u_filled, v_filled)

        # VectorWind
        vw = VectorWind(u_corr, v_corr)
        u_div, v_div = vw.divergentcomponent()
        v_div_da = xr.DataArray(v_div, coords=va_mean.coords, dims=va_mean.dims, name="v_div")

        # ----------------------------
        # Select sector and compute MSE
        # ----------------------------
        lon_min, lon_max = lon_slice.start, lon_slice.stop
        ta_sec = ta_mean.sel(lon=lon_slice)
        q_sec  = q_mean.sel(lon=lon_slice)
        z_sec  = z_mean.sel(lon=lon_slice)
        v_sec  = v_div_da.sel(lon=lon_slice)

        # Compute moist static energy
        Cp = 1004.0
        Lv = 2.5e6
        g  = 9.81
        h = Cp*ta_sec + Lv*q_sec + g*z_sec

        # Interpolate h to wind grid
        h_on_v = h.interp(lat=v_sec.lat, lon=v_sec.lon, plev=v_sec.plev, method="linear")

        
        # Meridional MSE flux
        vh = v_sec * h_on_v

        # Ensure monotonic plev and fill NaNs
        vh_sorted = vh.sortby("plev").fillna(0.0)
        plev_vals = np.array(vh_sorted.plev.values, dtype=float)
        order = np.argsort(plev_vals)
        vh_clean = vh_sorted.isel(plev=order).assign_coords(plev=plev_vals[order])

        # Vertical integration
        vh_int = vh_clean.integrate("plev")
        F_div = vh_int / g  # W/m
        #################################

        Fdiv_by_exp[exp] = F_div

    return Fdiv_by_exp

# def compute_Fdiv_for_model_experiments(
#     model_name,
#     model_meta,
#     experiments,
#     base_dir,
#     lon_slice=slice(0,360),
#     season=(1,12),
#     time_slice=None,   # slice(start_year, end_year)
# ):
#     """
#     Compute vertically-integrated divergent meridional MSE flux (F_div)
#     for all requested experiments of a given model.

#     Parameters
#     ----------
#     model_name : str
#         Model name (e.g., "UKESM1-0-LL")
#     model_meta : dict
#         Metadata for the model (institution, ensemble, grid)
#     experiments : dict
#         Experiment metadata dict
#     base_dir : Path
#         Base CEDA path
#     lon_slice : slice
#         Longitude range to select
#     season : tuple(int,int)
#         Start and end month for seasonal mean
#     time_slice : slice
#         Start and end years, e.g., slice(2071, 2101)

#     Returns
#     -------
#     Fdiv_by_exp : dict
#         Keys are experiment names, values are xarray.DataArray (lat, lon)
#     """

#     Fdiv_by_exp = {}

#     # List of variables we need
#     varnames = ["ta","hus","zg","ua","va"]

#     for exp, meta in experiments.items():

#         # # Only compute for scenario experiments
#         # if exp not in ["SSP245","G6solar","G6sulfur"]:
#         #     continue

#         # Ensemble overrides
#         if model_name == "CESM2-WACCM":
#             if meta["scenario"] == "G6sulfur":
#                 ensemble = "r1i1p1f2"
#             else:
#                 ensemble = "r1i1p1f1"
#         else:
#             ensemble = model_meta["ensemble"]

#         # ----------------------------
#         # Function to open & slice a variable safely
#         # ----------------------------
#         def open_var(varname):
#             base = (
#                 base_dir
#                 / meta["project"]
#                 / model_meta["institution"]
#                 / model_name
#                 / meta["scenario"]
#                 / ensemble
#                 / "Amon"
#                 / varname
#                 / model_meta["grid"]
#                 / "latest"
#             )

#             files = sorted(base.glob("*.nc"))
#             if len(files) == 0:
#                 raise FileNotFoundError(f"No NetCDF files found in {base}")

#             # Open with safe combine
#             ds = xr.open_mfdataset(
#                 [str(f) for f in files],
#                 combine="nested",
#                 concat_dim="time",
#                 decode_times=True,
#                 use_cftime=True,
#                 parallel=True,
#             )

#             # Read variable
#             da = mf.read_var(ds, varname)
#             if da is None:
#                 raise ValueError(f"Variable {varname} not found in {base}")

#             # Apply time slice if requested
#             if time_slice is not None:
#                 start_year, end_year = time_slice.start, time_slice.stop-1
#                 da = da.sel(time=year_slice_to_cftime(da, start_year, end_year))

#             return da

#         # ----------------------------
#         # Open all variables
#         # ----------------------------
#         ta  = open_var("ta")
#         q   = open_var("hus")
#         zg  = open_var("zg")
#         ua  = open_var("ua")
#         va  = open_var("va")

#         # ----------------------------
#         # Compute seasonal mean
#         # ----------------------------
#         start_month, end_month = season
#         ta_mean = mf.seasonal_mean_by_year(ta, start_month, end_month).mean("year")
#         q_mean  = mf.seasonal_mean_by_year(q,  start_month, end_month).mean("year")
#         z_mean  = mf.seasonal_mean_by_year(zg, start_month, end_month).mean("year")
#         ua_mean = mf.seasonal_mean_by_year(ua, start_month, end_month).mean("year")
#         va_mean = mf.seasonal_mean_by_year(va, start_month, end_month).mean("year")

#         # ----------------------------
#         # Compute divergent winds
#         # ----------------------------
#         # Fill NaNs
#         u_filled = ua_mean.fillna(0.0)
#         v_filled = va_mean.fillna(0.0)

#         # Barotropic mass correction
#         u_corr, v_corr = mf.barotropic_mass_correction(u_filled, v_filled)

#         # VectorWind
#         vw = VectorWind(u_corr, v_corr)
#         u_div, v_div = vw.divergentcomponent()
#         v_div_da = xr.DataArray(v_div, coords=va_mean.coords, dims=va_mean.dims, name="v_div")

#         # ----------------------------
#         # Select sector and compute MSE
#         # ----------------------------
#         lon_min, lon_max = lon_slice.start, lon_slice.stop
#         ta_sec = ta_mean.sel(lon=lon_slice)
#         q_sec  = q_mean.sel(lon=lon_slice)
#         z_sec  = z_mean.sel(lon=lon_slice)
#         v_sec  = v_div_da.sel(lon=lon_slice)

#         # Compute moist static energy
#         Cp = 1004.0
#         Lv = 2.5e6
#         g  = 9.81
#         h = Cp*ta_sec + Lv*q_sec + g*z_sec

#         # Interpolate h to wind grid
#         h_on_v = h.interp(lat=v_sec.lat, lon=v_sec.lon, plev=v_sec.plev, method="linear")

#         ######################################
#         import numpy as np
#         import xarray as xr
        
#         # Earth radius
#         a = 6.371e6
#         lat_rad = np.deg2rad(v_sec.lat)
        
#         # -------------------
#         # 1. Compute divergence of meridional wind (RHS1)
#         # -------------------
#         # meridional divergence: ∇·v = 1/(a cosφ) ∂(v cosφ)/∂φ
#         div_v = (1 / (a * np.cos(lat_rad))) * np.gradient(
#             v_sec * np.cos(lat_rad), lat_rad, axis=v_sec.get_axis_num("lat")
#         )
        
#         RHS1 = h_on_v * div_v
#         RHS1_int = RHS1.sortby("plev").integrate("plev") / g  # vertically integrated
        
#         # -------------------
#         # 2. Compute advection term (RHS2)
#         # -------------------
#         # meridional gradient of h
#         dh_dphi = np.gradient(h_on_v, lat_rad, axis=h_on_v.get_axis_num("lat"))
#         dh_dy = dh_dphi / a  # convert to m
        
#         RHS2 = v_sec * dh_dy
#         RHS2_int = RHS2.sortby("plev").integrate("plev") / g  # vertically integrated
        
#         # -------------------
#         # 3. Sum of both terms
#         # -------------------
#         RHS_total = RHS1_int + RHS2_int
        
#         # -------------------
#         # Optional: compare with direct flux divergence
#         # -------------------
#         F_flux = (v_sec * h_on_v).sortby("plev").integrate("plev") / g
#         F_flux_cos = F_flux * np.cos(lat_rad)
#         dFcos_dphi = np.gradient(F_flux_cos, lat_rad, axis=F_flux.get_axis_num("lat"))
#         F_divergence_check = dFcos_dphi / np.cos(lat_rad) / a  # same as ∇·(vh)
        
#         # Verify
#         np.allclose(RHS_total, F_divergence_check, atol=1e-8)  # should be True
        
                
#         ##################################
        
#         # # Meridional MSE flux
#         # vh = v_sec * h_on_v

#         # # Ensure monotonic plev and fill NaNs
#         # vh_sorted = vh.sortby("plev").fillna(0.0)
#         # plev_vals = np.array(vh_sorted.plev.values, dtype=float)
#         # order = np.argsort(plev_vals)
#         # vh_clean = vh_sorted.isel(plev=order).assign_coords(plev=plev_vals[order])

#         # # Vertical integration
#         # vh_int = vh_clean.integrate("plev")
#         # F_div = vh_int / g  # W/m
#         ##################################

#         Fdiv_by_exp[exp] = RHS_total #F_div

#     return Fdiv_by_exp
