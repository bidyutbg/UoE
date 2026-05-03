#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyhdf.SD import SD, SDC
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
import xesmf as xe
import numpy as np

def read_trmm_file(filepath):
    """
    Read a single TRMM 3B42 HDF4 file.
    Returns xarray DataArray with (lat, lon) in mm/hr.
    """
    hdf = SD(str(filepath), SDC.READ)

    precip = hdf.select('precipitation').get()  # shape (1440, 400) = (nlon, nlat)

    # Transpose to (nlat, nlon)
    precip = precip.T  # now (400, 1440)

    # TRMM 3B42 standard grid
    lat = np.arange(-49.875, 49.876, 0.25)   # 400 points
    lon = np.arange(-179.875, 179.876, 0.25)  # 1440 points

    da = xr.DataArray(
        precip,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon}
    )

    # Fill value
    fill_value = hdf.select('precipitation').attributes().get('_FillValue', -9999.9)
    da = da.where(da != fill_value)

    hdf.end()
    return da  # mm/hr


def load_trmm_season(years, season='SUM', trmm_base='/badc/trmm/data/TRMM_3B42'):
    """
    Load TRMM 3B42 data and compute seasonal mean for given years.

    Parameters:
    -----------
    years    : list of years e.g. range(1998, 2020)
    season   : 'SUM' (JJAS: months 6-9) or 'WIN' (NDJF: months 11,12,1,2) or 'ANN' (Jan-Dec: Annual Mean)
    trmm_base: base path to TRMM data

    Returns:
    --------
    seasonal_mean : xarray DataArray (lat, lon) in mm/day
    """
    trmm_base = Path(trmm_base)

    if season == 'SUM':
        target_months = [6, 7, 8, 9]
    elif season == 'WIN':
        target_months = [11, 12, 1, 2]
    elif season == 'ANN':
        target_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    else:
        raise ValueError("season must be 'SUM' or 'WIN' or 'ANN'")

    all_daily = []

    for year in years:
        print(f"  Processing year {year}...")

        # Generate all dates for this year
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')

        for date in dates:
            if date.month not in target_months:
                continue

            # Day of year (TRMM uses DOY folders)
            doy = date.strftime('%j')  # e.g. '001', '002'
            day_dir = trmm_base / str(year) / doy

            if not day_dir.exists():
                print(f"    Missing: {day_dir}")
                continue

            # Collect all 8 3-hourly files for this day
            date_str = date.strftime('%Y%m%d')
            hours = ['03', '06', '09', '12', '15', '18', '21', '00']

            # '00' belongs to next day's file naming
            daily_slices = []
            for hr in hours:
                if hr == '00':
                    # 00 UTC file is named with next day
                    next_date = date + pd.Timedelta(days=1)
                    next_doy  = next_date.strftime('%j')
                    next_dir  = trmm_base / str(next_date.year) / next_doy
                    fname = next_dir / f"3B42.{next_date.strftime('%Y%m%d')}.{hr}.7.HDF"
                else:
                    fname = day_dir / f"3B42.{date_str}.{hr}.7.HDF"

                if not fname.exists():
                    continue

                try:
                    da = read_trmm_file(fname)
                    daily_slices.append(da)
                except Exception as e:
                    print(f"    Error reading {fname}: {e}")
                    continue

            if len(daily_slices) > 0:
                # Daily mean from 3-hourly slices
                daily_mean = xr.concat(daily_slices, dim='time').mean('time')
                all_daily.append(daily_mean)

    if len(all_daily) == 0:
        raise ValueError(f"No TRMM data found for season {season}, years {list(years)}")

    # Stack all daily means and compute seasonal mean
    stacked       = xr.concat(all_daily, dim='time')
    seasonal_mean = stacked.mean('time') * 24  # mm/hr → mm/day

    return seasonal_mean


def load_trmm_seasonal_means(years=range(1998, 2020),
                              trmm_base='/badc/trmm/data/TRMM_3B42'):
    """
    Load TRMM JJAS and NDJF seasonal means.
    Returns dict with 'SUM' and 'WIN' and 'ANN' DataArrays in mm/day.
    """
    trmm_means = {}
    for season in ['SUM', 'WIN','ANN']:
        print(f"Loading TRMM {season}...")
        trmm_means[season] = load_trmm_season(years, season=season,
                                               trmm_base=trmm_base)
        print(f"TRMM {season} done: shape {trmm_means[season].shape}")
    return trmm_means




def regrid_to_common(da, target_lat=None, target_lon=None):
    """Regrid a DataArray to common lat/lon grid."""
    if target_lat is None:
        target_lat = np.arange(-50, 51, 2.5)
    elif isinstance(target_lat, tuple):
        target_lat = np.arange(target_lat[0], target_lat[1]+1, 2.5)
    elif isinstance(target_lat, (int, float)):
        target_lat = np.arange(-target_lat, target_lat+1, 2.5)

    if target_lon is None:
        target_lon = np.arange(0, 360, 2.5)  # always default to full lon
    elif isinstance(target_lon, tuple):
        target_lon = np.arange(target_lon[0], target_lon[1]+1, 2.5)

    ds_out = xr.Dataset({
        'lat': (['lat'], target_lat),
        'lon': (['lon'], target_lon)
    })

    ds_in = da.to_dataset(name='data')
    regridder = xe.Regridder(ds_in, ds_out, method='bilinear', reuse_weights=False)
    return regridder(da) 


# In[ ]:


# Usage
trmm_means = load_trmm_seasonal_means(years=range(1998, 2020))


# In[ ]:


def compute_monsoon_mask(mean_summer, mean_winter, mean_annual,
                          precip_threshold=2.0, summer_fraction=0.55):
    """
    Global monsoon domain:
    - NH (lat>0): wet=JJAS, dry=NDJF
    - SH (lat<0): wet=NDJF, dry=JJAS
    Conditions:
    - wet_mean - dry_mean >= precip_threshold (mm/day)
    - wet_total / ann_total >= summer_fraction
    """
    s   = mean_summer ;#* 86400  # mm/day
    w   = mean_winter ;#* 86400
    ann = mean_annual ;#* 86400

    lat = mean_summer.lat

    # Wet and dry season means depending on hemisphere
    wet_mean = xr.where(lat > 0, s, w)   # JJAS for NH, NDJF for SH
    dry_mean = xr.where(lat > 0, w, s)   # NDJF for NH, JJAS for SH

    # Totals
    wet_total = wet_mean * 4
    ann_total = ann * 12

    cond1 = (wet_mean - dry_mean) >= precip_threshold
    cond2 = (wet_total / ann_total.where(ann_total > 0)) >= summer_fraction

    return (cond1 & cond2).astype(float)


# In[ ]:


mean_summer_trmm = trmm_means['SUM']
mean_winter_trmm = trmm_means['WIN']
mean_annual_trmm = trmm_means['ANN']

monsoon_masks_trmm = compute_monsoon_mask(mean_summer_trmm, mean_winter_trmm, mean_annual_trmm)


# In[ ]:


monsoon_masks_trmm.to_netcdf('./TRMM_monsoon_contour.nc')

