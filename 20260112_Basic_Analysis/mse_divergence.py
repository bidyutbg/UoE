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

        # ----------------------------
        # Select sector and compute MSE
        # ----------------------------
        lon_min, lon_max = lon_slice.start, lon_slice.stop
        ta_sec = ta_mean.sel(lon=lon_slice)
        q_sec  = q_mean.sel(lon=lon_slice)
        z_sec  = z_mean.sel(lon=lon_slice)
        u_sec  = u_filled.sel(lon=lon_slice)
        v_sec  = v_filled.sel(lon=lon_slice)

        # Compute moist static energy
        Cp = 1004.0
        Lv = 2.5e6
        g  = 9.81
        h = Cp*ta_sec + Lv*q_sec + g*z_sec

        # Interpolate h to wind grid
        h_on_v = h.interp(lat=v_sec.lat, lon=v_sec.lon, plev=v_sec.plev, method="linear")

        # Zonal wind * MSE
        uh = u_sec * h_on_v
        # Meridional wind * MSE
        vh = v_sec * h_on_v

        # Ensure monotonic plev and fill NaNs
        uh_sorted = uh.sortby("plev").fillna(0.0)
        vh_sorted = vh.sortby("plev").fillna(0.0)
        
        plev_vals = np.array(vh_sorted.plev.values, dtype=float)
        order = np.argsort(plev_vals)

        uh_clean = uh_sorted.isel(plev=order).assign_coords(plev=plev_vals[order])
        vh_clean = vh_sorted.isel(plev=order).assign_coords(plev=plev_vals[order])
        
        # Vertical integration
        uh_int = uh_clean.integrate("plev") / g
        vh_int = vh_clean.integrate("plev") / g


        vw = VectorWind(uh_int, vh_int)
        Flambda_div, Fphi_div = vw.divergentcomponent()

        Fphi_div = xr.DataArray(
        Fphi_div,                       # the numpy array
        coords={"lat": uh_int.lat, 
                "lon": uh_int.lon},
        dims=("lat", "lon"),
        name="Fphi_div"
        )
        Fphi_div.attrs["long_name"] = "divergent meridional MSE flux"
        Fphi_div.attrs["units"] = "W m^-1"


        Fdiv_by_exp[exp] = Fphi_div

    return Fdiv_by_exp

    #     #Compute div of v_div (divergence of divergent component of v)
    #     # Earth radius
    #     a = 6.371e6
        
    #     lat_rad = np.deg2rad(v_sec.lat)
    #     coslat = xr.DataArray(
    #         np.cos(lat_rad),
    #         coords={"lat": v_sec.lat},
    #         dims=("lat",)
    #     )
        
    #     # d/dφ (v cosφ)
    #     d_vcos_dphi = (v_sec * coslat).differentiate("lat")
        
    #     # ∇·v_div
    #     div_v = d_vcos_dphi / (a * coslat)

    #     # Compute moist static energy
    #     Cp = 1004.0
    #     Lv = 2.5e6
    #     g  = 9.81
    #     h = Cp*ta_sec + Lv*q_sec + g*z_sec

    #     # Interpolate h to wind grid
    #     h_on_v = h.interp(lat=v_sec.lat, lon=v_sec.lon, plev=v_sec.plev, method="linear")

        
    #     # Meridional MSE flux
    #     vh = h_on_v * div_v

    #     # Ensure monotonic plev and fill NaNs
    #     vh_sorted = vh.sortby("plev").fillna(0.0)
    #     plev_vals = np.array(vh_sorted.plev.values, dtype=float)
    #     order = np.argsort(plev_vals)
    #     vh_clean = vh_sorted.isel(plev=order).assign_coords(plev=plev_vals[order])

    #     # Vertical integration
    #     vh_int = vh_clean.integrate("plev")
    #     RHS_1 = vh_int / g  # W/m = RHS_1
    #     #################################

    #     #Compute RHS_2 d_div*(1/a)(del h/ del phi); phi is latitude
    #     # dh_dphi = np.gradient(h_on_v, lat, axis=h_on_v.get_axis_num("lat"))
    #     # dh_dy = dh_dphi / a
        
    #     # d/dφ (h cosφ)
    #     d_hcos_dphi = (h_on_v * coslat).differentiate("lat")
        
    #     # (1/a) ∂h/∂y with spherical consistency
    #     dh_dy = (d_hcos_dphi / coslat) / a
        
    #     # v · ∇h  (meridional only)
    #     h_adv = v_sec * dh_dy

    #     # Ensure monotonic plev and fill NaNs
    #     h_adv_sorted = h_adv.sortby("plev").fillna(0.0)
    #     # plev_vals = np.array(vh_sorted.plev.values, dtype=float)
    #     # order = np.argsort(plev_vals)
    #     h_adv_clean = h_adv_sorted.isel(plev=order).assign_coords(plev=plev_vals[order])

    #     # Vertical integration
    #     h_adv_int = h_adv_clean.integrate("plev")
    #     RHS_2= h_adv_int / g
        


    #     Fdiv_by_exp[exp] = RHS_1 ;#+ RHS_2

    # return Fdiv_by_exp

import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind

import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind

def barotropic_correction(u, v, plev, ps):
    """
    Barotropic mass correction following Trenberth (1991)
    and Wei & Bordoni (2020) Supplementary Text S1.

    Parameters
    ----------
    u, v : xr.DataArray
        Zonal and meridional winds (plev, lat, lon)
    plev : xr.DataArray
        Pressure levels in Pa (plev,)
    ps : xr.DataArray
        Surface pressure (lat, lon)

    Returns
    -------
    u_corr, v_corr : xr.DataArray
        Barotropically corrected winds
    """

    # -----------------------------
    # 1. Compute layer thickness dp
    # -----------------------------
    dp = plev.diff("plev")
    dp = xr.concat([dp.isel(plev=0), dp], dim="plev")
    dp = dp.assign_coords(plev=plev)

    # -------------------------------------------
    # 2. Vertically integrated mass flux M = ∫ v dp
    # -------------------------------------------
    Mx = (u * dp).sum("plev")  # zonal mass flux
    My = (v * dp).sum("plev")  # meridional mass flux

    Mx = Mx.fillna(0.0)
    My = My.fillna(0.0)

    # ------------------------------------------------------------
    # 3. Extract DIVERGENT (irrotational) component of M using:
    #       M_div = ∇χ = vw.irrotationalcomponent()
    # ------------------------------------------------------------
    vwM = VectorWind(Mx, My)
    Mx_div, My_div = vwM.irrotationalcomponent()

    # ------------------------------------------------------------
    # 4. Barotropic correction: M_bt = - M_div / p_s
    #
    # Because we want:
    #    M + p_s * u_bt = M_nondivergent
    # => p_s * u_bt = - M_div
    # ------------------------------------------------------------
    u_bt = -Mx_div / ps
    v_bt = -My_div / ps

    # ------------------------------------------------------------
    # 5. Expand barotropic correction to all vertical levels
    # ------------------------------------------------------------
    u_bt_3D = u_bt.expand_dims({"plev": u.plev}, axis=0)
    v_bt_3D = v_bt.expand_dims({"plev": v.plev}, axis=0)

    u_corr = u + u_bt_3D
    v_corr = v + v_bt_3D

    return u_corr, v_corr


import numpy as np
import xarray as xr

def make_constant_ps_like(sample_da, value_hPa=1000.0):
    """
    Create a constant surface-pressure DataArray on the (lat, lon) grid of 'sample_da'.

    Parameters
    ----------
    sample_da : xr.DataArray or xr.Dataset
        Any object that contains 'lat' and 'lon' coordinates on the target grid.
        Typically use u.isel(plev=0) or v.isel(plev=0) or any 2-D field with (lat, lon).
    value_hPa : float
        Constant surface pressure in hPa (default: 1000 hPa).

    Returns
    -------
    ps : xr.DataArray
        Surface pressure (Pa) with dims ('lat', 'lon') on the same grid as sample_da.
    """
    # get coords from the sample
    lat = sample_da['lat']
    lon = sample_da['lon']

    ps_values = np.full((lat.size, lon.size), value_hPa * 100.0, dtype=np.float32)  # convert hPa -> Pa
    ps = xr.DataArray(
        ps_values,
        coords={'lat': lat, 'lon': lon},
        dims=('lat', 'lon'),
        name='ps'
    )
    ps.attrs.update({'units': 'Pa', 'long_name': f'Constant surface pressure ({value_hPa:.0f} hPa)'})
    return ps
