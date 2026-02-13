import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind

def barotropic_correction(u, v, plev, ps):
    """
    Implements the barotropic mass-correction described in
    Wei & Bordoni (2020) Supplementary Information, Text S1.

    Parameters
    ----------
    u : xarray.DataArray
        Zonal wind, dims = (plev, lat, lon), units: m s-1
    v : xarray.DataArray
        Meridional wind, dims = (plev, lat, lon), units: m s-1
    plev : xarray.DataArray
        Pressure levels (Pa), dims = (plev,)
    ps : xarray.DataArray
        Surface pressure (Pa), dims = (lat, lon)
    
    Returns
    -------
    u_corr : xarray.DataArray
        Barotropically corrected zonal wind (same dims as u)
    v_corr : xarray.DataArray
        Barotropically corrected meridional wind (same dims as v)
    """

    # ============================================================
    # 1. Compute layer thickness Δp (Pa) for mass weighting
    # ============================================================
    dp = plev.diff("plev")
    dp = xr.concat([dp.isel(plev=0), dp], dim="plev")
    dp = dp.assign_coords(plev=plev)

    # ============================================================
    # 2. Compute vertically integrated mass flux M_x, M_y
    #    M = ∫ u dp ,  ∫ v dp          (Pa·m/s)
    # ============================================================
    Mx = (u * dp).sum("plev")
    My = (v * dp).sum("plev")

    # Make sure no NaNs break spectral calculations
    Mx = Mx.fillna(0.0)
    My = My.fillna(0.0)

    # ============================================================
    # 3. Compute divergence D = ∇·M   (Pa·m/s / m)
    #    Using spherical geometry (windspharm handles it)
    # ============================================================
    vw = VectorWind(Mx, My)
    D = vw.divergence()   # ∇·(Mx, My)

    # ============================================================
    # 4. Solve ∇² Φ = -D  (Poisson equation on the sphere)
    #    windspharm calls spherical harmonics internally
    # ============================================================
    # NOTE: The "irrotationalcomponent" of a vector field = ∇χ.
    # For a scalar Poisson problem, we use "scalar_potential".
    Phi = vw.scalar_potential()   # solves ∇² Phi = divergence(M)

    # So to solve ∇²Φ = -D, simply negate the potential:
    Phi = -Phi

    # ============================================================
    # 5. Compute corrected, vertically-uniform barotropic winds
    #    p_s (u*, v*) = ∇Φ
    #    u* = Φ_λ / (a cosφ p_s)
    #    v* = Φ_φ / (a p_s)
    # ============================================================
    a = 6.371e6  # Earth radius (m)
    lat_rad = np.deg2rad(u["lat"])

    # Compute gradient of Phi
    vw_p = VectorWind(Phi, Phi)  # Dummy second arg not used
    dPhi_dlambda, dPhi_dphi = vw_p.gradient()  # ∂Φ/∂λ, ∂Φ/∂φ

    # Avoid division by zero at poles
    cosphi = np.cos(lat_rad)
    cosphi = xr.where(np.abs(cosphi) < 1e-6, 1e-6, cosphi)

    u_bt = (1.0 / (a * cosphi * ps)) * dPhi_dlambda
    v_bt = (1.0 / (a * ps))          * dPhi_dphi

    # ============================================================
    # 6. Add barotropic correction to ALL pressure levels
    # ============================================================
    # Expand bt winds into vertical levels
    u_bt_3D = u_bt.expand_dims({"plev": u.plev}, axis=0)
    v_bt_3D = v_bt.expand_dims({"plev": v.plev}, axis=0)

    u_corr = u + u_bt_3D
    v_corr = v + v_bt_3D

    return u_corr, v_corr