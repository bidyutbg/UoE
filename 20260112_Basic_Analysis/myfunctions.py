import numpy as np
import xarray as xr
from pathlib import Path
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from shapely.prepared import prep
import cftime


# =========================
# Functions
# =========================
def open_files(path):
    """Open all NetCDF files in a directory with xarray"""
    return xr.open_mfdataset(
    f"{path}/*.nc",
    combine="nested",
    concat_dim="time",
    decode_times=True,
    use_cftime=True,
    parallel=True,)


def open_files_CESM_G6sulfur(path):
    """
    Open only the relevant CESM2-WACCM G6sulfur files.
    Reads only the 202001-206912 and 207001-210012 files.
    """
    path = Path(path)

    # Expand the two patterns
    files_to_read = list(path.glob("*_202001-206912.nc")) + list(path.glob("*_207001-210012.nc"))
    
    # Open only these files with xarray
    return xr.open_mfdataset([str(f) for f in files_to_read], combine="by_coords", parallel=True)


def open_files_CESM_ssp585(path):
    """
    Open only the relevant CESM2-WACCM ssp585 files.
    Reads only the 201501-210012 files.
    """
    path = Path(path)

    # Define the exact files you want
    files_to_read = list(path.glob("*_201501-210012.nc"))
    # Open only these files with xarray
    return xr.open_mfdataset([str(f) for f in files_to_read], combine="by_coords", parallel=True)


def open_files_IPSL_ssp585(path):
    """
    Open only the relevant IPSL-CM6A-LR ssp585 files.
    Reads only the 201501-210012 files.
    """
    path = Path(path)

    # Define the exact files you want
    files_to_read = list(path.glob("*_201501-210012.nc"))
    
    # Open only these files with xarray
    return xr.open_mfdataset([str(f) for f in files_to_read], combine="by_coords", parallel=True)


def read_var(ds, var):
    """Extract a variable from dataset"""
    return ds[var]



def year_slice_to_cftime(da, start_year, end_year):
    """
    Convert integer years to cftime slice for a DataArray
    """
    # Assume 360-day calendar; could check da.time.encoding["calendar"]
    start = cftime.Datetime360Day(start_year, 1, 1)
    end   = cftime.Datetime360Day(end_year, 12, 30)
    return slice(start, end)



def get_vertical_dim(da):
    for d in ["lev", "plev"]:
        if d in da.dims:
            return d
    raise ValueError(f"No vertical dimension found in {da.dims}")

def get_lon_dim(da):
    for d in ["lon", "longitude"]:
        if d in da.dims:
            return d
    raise ValueError(f"No longitude dimension found in {da.dims}")

def climatological_mean(da, start_month, end_month):
    """
    Compute climatological mean over all years
    for a given month range.
    
    Handles cross-year seasons (e.g. DJF).
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data with time dimension
    start_month : int
        Starting month (1–12)
    end_month : int
        Ending month (1–12)
        
    Returns
    -------
    xarray.DataArray
        Climatological mean
    """

    month = da["time"].dt.month

    if start_month <= end_month:
        # Normal season (e.g. JJAS)
        da_sel = da.where(
            (month >= start_month) & (month <= end_month),
            drop=True
        )
    else:
        # Cross-year season (e.g. DJF)
        da_sel = da.where(
            (month >= start_month) | (month <= end_month),
            drop=True
        )

    return da_sel.mean("time")


import xarray as xr

def seasonal_mean_by_year_old(da, start_month, end_month):
    """
    Compute seasonal mean for each year.
    Handles cross-year seasons (e.g. DJF) correctly.
    """

    month = da["time"].dt.month
    year = da["time"].dt.year

    if start_month <= end_month:
        # Normal season (e.g. JJAS)
        da_sel = da.where(
            (month >= start_month) & (month <= end_month),
            drop=True
        )

        da_season = da_sel.groupby("time.year").mean("time")

    else:
        # Cross-year season (e.g. DJF)
        da_sel = da.where(
            (month >= start_month) | (month <= end_month),
            drop=True
        )

        # Recompute month/year AFTER selection
        month_sel = da_sel["time"].dt.month
        year_sel = da_sel["time"].dt.year

        season_year = xr.where(month_sel == 12, year_sel + 1, year_sel)

        da_season = (
            da_sel
            .assign_coords(season_year=("time", season_year.data))
            .groupby("season_year")
            .mean("time")
            .rename({"season_year": "year"})
        )

    return da_season

def seasonal_mean_by_year_old2(da, start_month, end_month):
    """
    Compute seasonal mean for each year.
    Handles cross-year seasons (e.g., DJF) correctly.
    Works with both datetime64 and cftime calendars.
    """
    # Extract month and year
    try:
        month = da["time"].dt.month
        year  = da["time"].dt.year
    except AttributeError:
        # fallback for cftime: convert to pandas index
        idx = da["time"].to_index()
        month = idx.month
        year  = idx.year

    if start_month <= end_month:
        # Normal season (e.g., JJAS)
        mask = xr.DataArray(
            (month >= start_month) & (month <= end_month),
            coords={"time": da["time"]},
            dims="time"
        )
        da_sel = da.where(mask, drop=True)
        # Create explicit 'year_coord' for grouping
        year_coord = xr.DataArray(year[mask.values], coords={"time": da_sel["time"]}, dims="time")
        da_season = da_sel.groupby(year_coord).mean("time").rename({"year_coord": "year"})

    else:
        # Cross-year season (e.g., DJF)
        mask = xr.DataArray(
            (month >= start_month) | (month <= end_month),
            coords={"time": da["time"]},
            dims="time"
        )
        da_sel = da.where(mask, drop=True)

        # Recompute month/year after selection
        try:
            month_sel = da_sel["time"].dt.month
            year_sel  = da_sel["time"].dt.year
        except AttributeError:
            idx_sel = da_sel["time"].to_index()
            month_sel = idx_sel.month
            year_sel  = idx_sel.year

        # Assign December to next year
        season_year = xr.DataArray(
            np.where(month_sel == 12, year_sel + 1, year_sel),
            coords={"time": da_sel["time"]},
            dims="time"
        )

        da_season = da_sel.groupby(season_year).mean("time").rename({"season_year": "year"})

    return da_season

def seasonal_mean_by_year(da, start_month, end_month):
    """
    Compute seasonal mean for each year.
    Handles cross-year seasons (e.g., DJF) correctly.
    Works with cftime calendars.
    """

    # Extract month and year (convert to integers for cftime)
    try:
        month = da["time"].dt.month
        year  = da["time"].dt.year
    except AttributeError:
        # cftime index fallback
        time_index = da["time"].to_index()
        month = xr.DataArray(time_index.month, coords={"time": da["time"]}, dims="time")
        year  = xr.DataArray(time_index.year, coords={"time": da["time"]}, dims="time")

    if start_month <= end_month:
        # Normal season (e.g., JJAS)
        mask = (month >= start_month) & (month <= end_month)
        da_sel = da.where(mask, drop=True)

        # Group by year and take mean
        da_season = da_sel.groupby(year).mean("time")
        da_season = da_season.rename({year.name: "year"})

    else:
        # Cross-year season (e.g., DJF)
        mask = (month >= start_month) | (month <= end_month)
        da_sel = da.where(mask, drop=True)

        # Correct year for December
        month_sel = da_sel["time"].dt.month if hasattr(da_sel["time"], "dt") else da_sel["time"].to_index().month
        year_sel  = da_sel["time"].dt.year  if hasattr(da_sel["time"], "dt") else da_sel["time"].to_index().year
        year_coord = xr.DataArray(
            np.where(month_sel == 12, year_sel + 1, year_sel),
            coords={"time": da_sel["time"]},
            dims="time",
        )

        # Group by corrected year
        da_season = da_sel.groupby(year_coord).mean("time")
        da_season = da_season.rename({year_coord.name: "year"})

    return da_season



def mask_land_cartopy(da):
    land_geom = prep(cfeature.NaturalEarthFeature(
        "physical", "land", "110m"
    ).geometries().__next__())

    lon2d, lat2d = np.meshgrid(da.longitude, da.latitude)

    mask = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lon2d.shape[0]):
        for j in range(lon2d.shape[1]):
            mask[i, j] = land_geom.contains(
                sgeom.Point(lon2d[i, j], lat2d[i, j])
            )

    return da.where(~mask)

def stipple_nonsignificant(ax, lon, lat, mask, stride=3):
    """
    mask = True where NOT significant
    Works for (lat, lon) or (lon, lat)
    """

    # Build 2D coordinate grids safely
    Lon, Lat = np.meshgrid(lon.values, lat.values)

    # Align mask to (lat, lon)
    if mask.dims != ("latitude", "longitude"):
        mask = mask.transpose("latitude", "longitude")

    jj, ii = np.where(mask.values)

    ax.scatter(
        Lon[jj, ii][::stride],
        Lat[jj, ii][::stride],
        s=2,
        color="k",
        alpha=0.4,
        transform=ccrs.PlateCarree(),
        zorder=20,
        rasterized=True
    )

def barotropic_mass_correction(u, v, lat_dim='lat', lon_dim='lon', plev_dim='plev'):
    # Compute barotropic (vertically averaged) wind
    u_barotropic = u.mean(dim=plev_dim)
    v_barotropic = v.mean(dim=plev_dim)
    
    # Subtract barotropic component from each level
    u_corr = u - u_barotropic
    v_corr = v - v_barotropic
    return u_corr, v_corr


