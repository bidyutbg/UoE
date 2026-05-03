#!/usr/bin/env python
# coding: utf-8

# """
# Rainy season onset/demise from TRMM 3B42 daily observations (1998–2019).
# 
# Adapted from the CMIP6 model version (Bombardi et al. 2019 method).
# TRMM reader functions are taken from the companion trmm_utils module.
# 
# Dependencies: numpy, xarray, pyhdf (SD/SDC), pandas, math, pathlib
# """

# In[1]:


import os
import sys
import math
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from pyhdf.SD import SD, SDC


sys.path.append('./rainyseason_functions/')
from rainyseason_onset import rainyseason_onset
from rainyseason_B17_onset import rainyseason_B17_onset
from rainyseason_demise import rainyseason_demise
from rainyseason_B17_demise import rainyseason_B17_demise

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# ── Configuration ─────────────────────────────────────────────────────────────

TRMM_BASE = Path("/badc/trmm/data/TRMM_3B42")
OUT_DIR   = Path("./rainyseason_output_files")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START = 1998
YEAR_END   = 2019

varname  = "pr"
missval  = -999.0
dper     = 25.0
npass    = 50

# TRMM 3B42 uses a standard 365/366-day Gregorian calendar.
# We work with DOY 1-365 throughout (leap-day data are included in the
# daily means but DOY is capped at 365 for the harmonic/cycle arrays).
TOT_INT = 365   # fixed harmonic cycle length


# In[ ]:


# ── TRMM reader  ─────────────────────────────────────────────

def read_trmm_file(filepath):
    """
    Read a single TRMM 3B42 HDF4 file.
    Returns xarray DataArray (lat, lon) in mm/hr.
    """
    hdf    = SD(str(filepath), SDC.READ)
    precip = hdf.select("precipitation").get()   # (nlon=1440, nlat=400)
    precip = precip.T                            # → (400, 1440)

    lat = np.arange(-49.875, 49.876, 0.25)      # 400 points
    lon = np.arange(-179.875, 179.876, 0.25)    # 1440 points

    da = xr.DataArray(precip, dims=["lat", "lon"],
                      coords={"lat": lat, "lon": lon})

    fill_value = hdf.select("precipitation").attributes().get("_FillValue", -9999.9)
    da = da.where(da != fill_value)
    hdf.end()
    return da   # mm/hr


def load_trmm_daily(year_start, year_end, trmm_base):
    """
    Load TRMM 3B42 and return a daily DataArray (time, lat, lon) in mm/day.

    Each calendar day is assembled from up to 8 three-hourly files
    (03, 06, 09, 12, 15, 18, 21 UTC of that day + 00 UTC of the next day).
    The daily mean in mm/hr is then multiplied by 24 to give mm/day.
    """
    trmm_base  = Path(trmm_base)
    hours      = ["03", "06", "09", "12", "15", "18", "21", "00"]
    all_daily  = []
    all_dates  = []

    for year in range(year_start, year_end + 1):
        print(f"  Loading TRMM year {year}...")
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

        for date in dates:
            doy     = date.strftime("%j")
            day_dir = trmm_base / str(year) / doy
            date_str = date.strftime("%Y%m%d")

            daily_slices = []
            for hr in hours:
                if hr == "00":
                    next_date = date + pd.Timedelta(days=1)
                    next_doy  = next_date.strftime("%j")
                    next_dir  = trmm_base / str(next_date.year) / next_doy
                    fname = next_dir / f"3B42.{next_date.strftime('%Y%m%d')}.{hr}.7.HDF"
                else:
                    fname = day_dir / f"3B42.{date_str}.{hr}.7.HDF"

                if not fname.exists():
                    continue
                try:
                    daily_slices.append(read_trmm_file(fname))
                except Exception as e:
                    print(f"    Error reading {fname.name}: {e}")

            if daily_slices:
                daily_mean = xr.concat(daily_slices, dim="slot").mean("slot")
                all_daily.append(daily_mean * 24.0)   # mm/hr → mm/day
                all_dates.append(date)

    if not all_daily:
        raise ValueError("No TRMM data found — check TRMM_BASE path.")

    time_index = pd.DatetimeIndex(all_dates)
    da = xr.concat(all_daily, dim=pd.Index(time_index, name="time"))
    da.attrs["units"] = "mm/day"
    return da


# ── Helper functions (identical to model script) ──────────────────────────────

def Harmonics(coefa, coefb, hvar, tseries, nmodes, missval):
    mtot  = len(tseries)
    time  = np.arange(1, mtot + 1, 1.)
    tdata = tseries.copy()
    tdata[tseries == missval] = 0.
    svar  = sum((tdata - np.mean(tdata))**2) / (mtot - 1)
    nm    = nmodes
    if 2 * nm > mtot:
        nm = mtot // 2
    coefa = np.zeros(nm); coefb = np.zeros(nm); hvar = np.zeros(nm)
    for tt in range(nm):
        Ak = np.sum(tdata * np.cos(2. * math.pi * (tt+1) * time / float(mtot)))
        Bk = np.sum(tdata * np.sin(2. * math.pi * (tt+1) * time / float(mtot)))
        coefa[tt] = Ak * 2. / float(mtot)
        coefb[tt] = Bk * 2. / float(mtot)
        hvar[tt]  = mtot * (coefa[tt]**2 + coefb[tt]**2) / (2. * (mtot-1) * svar)
    return coefa, coefb, hvar


def build_time_arrays_pandas(time_coord):
    """Build jday/day/month/year arrays from a pandas DatetimeIndex time coord."""
    dti   = pd.DatetimeIndex(time_coord.values)
    year  = dti.year.values.astype(int)
    month = dti.month.values.astype(int)
    day   = dti.day.values.astype(int)
    # Cap DOY at 365 so leap days (DOY 366) map to DOY 365.
    # This keeps the harmonic cycle arrays at a fixed TOT_INT=365.
    jday  = np.minimum(dti.day_of_year.values.astype(int), TOT_INT)
    return jday, day, month, year


def make_var(data, long_name, units, nyrs_save):
    return xr.DataArray(
        data[:nyrs_save].astype(np.float32),
        dims=["year", "lat", "lon"],
        attrs={"long_name": long_name, "units": units, "missing_value": missval},
    )


# ── Output filename ───────────────────────────────────────────────────────────

outfile = OUT_DIR / f"rainy_season_TRMM_{YEAR_START}-{YEAR_END}.nc"

if outfile.exists():
    print(f"SKIP {outfile.name} — already exists")
else:

    # ── Load all TRMM daily data ──────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  TRMM  {YEAR_START}-{YEAR_END}")
    print(f"{'='*60}")

    da_mm = load_trmm_daily(YEAR_START, YEAR_END, TRMM_BASE)
    da_mm = da_mm.transpose("time", "lat", "lon")
    prec  = da_mm.values

    ntot = prec.shape[0]
    nlat = prec.shape[1]
    nlon = prec.shape[2]
    lats = da_mm.lat.values
    lons = da_mm.lon.values

    # ── Date arrays ───────────────────────────────────────────────────────────

    jday, day, month, year = build_time_arrays_pandas(da_mm.time)
    tot_int = TOT_INT          # 365 (fixed)
    tot     = float(tot_int)
    yr0     = int(year[0])
    nyrs    = int(year.max() - year.min()) + 1

    print(f"  tot={tot_int}  ntot={ntot}  nlat={nlat}  nlon={nlon}  nyrs={nyrs}")

    # ── Step 3: mean annual cycle + harmonics + startwet ─────────────────────

    prec[prec < 0.] = missval
    thres = ntot - 0.01 * dper * ntot
    mask  = np.zeros((nlat, nlon))
    for it in range(nlat):
        for jt in range(nlon):
            id = np.where(prec[:, it, jt] != missval)
            if len(id[0]) >= thres:
                mask[it, jt] = 1.

    rm = np.zeros((nlat, nlon))
    for it in range(nlat):
        for jt in range(nlon):
            if mask[it, jt] == 1.:
                tmp = prec[:, it, jt]
                id  = np.where(tmp >= 0.)
                if len(id[0]) > 1:
                    rm[it, jt] = np.mean(tmp[id[0]])

    cycle = np.zeros((tot_int, nlat, nlon))
    for tt in range(tot_int):
        id = np.where(jday == tt + 1)
        for it in range(nlat):
            for jt in range(nlon):
                if mask[it, jt] == 1.:
                    tmp = prec[id[0], it, jt]
                    id2 = np.where(tmp >= 0.)
                    if len(id2[0]) > 1:
                        cycle[tt, it, jt] = np.mean(tmp[id2[0]])

    time_arr  = np.arange(1, tot_int + 1, 1.)
    harm1     = np.zeros((nlat, nlon))
    harm2     = np.zeros((nlat, nlon))
    harm3     = np.zeros((nlat, nlon))
    harmonic1 = np.zeros((tot_int, nlat, nlon))
    smoothed  = np.zeros((tot_int, nlat, nlon))

    for it in range(nlat):
        for jt in range(nlon):
            if mask[it, jt] == 1.:
                coefa = np.zeros(3); coefb = np.zeros(3); hvar = np.zeros(3)
                coefa, coefb, hvar = Harmonics(coefa, coefb, hvar,
                                               cycle[:, it, jt], 3, missval)
                harm1[it, jt] = hvar[0]
                harm2[it, jt] = hvar[1]
                harm3[it, jt] = hvar[2]
                harmonic1[:, it, jt] = rm[it, jt]
                harmonic1[:, it, jt] += (
                    coefa[0] * np.cos(2.*math.pi*time_arr/tot)
                  + coefb[0] * np.sin(2.*math.pi*time_arr/tot))
                smoothed[:, it, jt] = np.mean(cycle[:, it, jt])
                for pp in range(3):
                    smoothed[:, it, jt] = (smoothed[:, it, jt]
                        + coefa[pp] * np.cos(2.*math.pi*time_arr*(pp+1)/float(tot_int))
                        + coefb[pp] * np.sin(2.*math.pi*time_arr*(pp+1)/float(tot_int)))

    id = np.where(harm2 >= harm1); mask[id] = 0.; rm[id] = 0.
    id = np.where(harm3 >= harm1); mask[id] = 0.; rm[id] = 0.

    startwet = np.zeros((nlat, nlon))
    for it in range(nlat):
        for jt in range(nlon):
            if mask[it, jt] == 1.:
                id = np.where(harmonic1[:, it, jt] == harmonic1[:, it, jt].min())
                startwet[it, jt] = jday[id[0][0]]

    print(f"  Step 3 done. Active grid points: {int(mask.sum())}/{nlat*nlon}")

    # ── Step 4: onset and demise ──────────────────────────────────────────────

    onset_jday   = np.zeros((nyrs, nlat, nlon))
    onset_day    = np.zeros((nyrs, nlat, nlon))
    onset_month  = np.zeros((nyrs, nlat, nlon))
    onset_year   = np.zeros((nyrs, nlat, nlon))
    demise_jday  = np.zeros((nyrs, nlat, nlon))
    demise_day   = np.zeros((nyrs, nlat, nlon))
    demise_month = np.zeros((nyrs, nlat, nlon))
    demise_year  = np.zeros((nyrs, nlat, nlon))

    wjd = np.zeros(nyrs); wd = np.zeros(nyrs)
    wm  = np.zeros(nyrs); wy = np.zeros(nyrs)
    wsc = np.zeros((nyrs, tot_int // 2))
    djd = np.zeros(nyrs); dd = np.zeros(nyrs)
    dm  = np.zeros(nyrs); dy = np.zeros(nyrs)
    dsc = np.zeros((nyrs, tot_int // 2))
    ap  = np.zeros(ntot)

    prec[prec < 0.] = 0.   # VERY IMPORTANT

    print("  Step 4: onset/demise loop...")
    for it in range(nlat):
        if it % 20 == 0:
            print(f"    lat {it+1}/{nlat}")
        for jt in range(nlon):
            if rm[it, jt] > 0.:
                sdate = startwet[it, jt]
                ap[:] = prec[:, it, jt] - rm[it, jt]

                sid     = np.where(jday == sdate)[0]
                sjday_  = jday[sid];  sday_   = day[sid]
                smonth_ = month[sid]; syear_  = year[sid]

                # ONSET first pass
                wjd[:]=0.; wd[:]=0.; wm[:]=0.; wy[:]=0.; wsc[:]=0.
                wjd[:],wd[:],wm[:],wy[:],wsc[:,:] = rainyseason_onset(
                    nyrs, tot_int, jday, day, month, year,
                    sdate, ap, wjd, wd, wm, wy, wsc)
                miss = np.where(wjd==0.); id = np.where(wjd!=0.)
                if len(id[0]) > 0:
                    tmpx = np.cos(wjd*math.pi/183.); tmpy = np.sin(wjd*math.pi/183.)
                    med  = math.atan2(np.median(tmpy[id]),np.median(tmpx[id]))*183./math.pi
                    if med < 0.: med += tot
                    tmpc = wjd[:] - med
                    if len(miss[0]) > 0: tmpc[miss] = 0.
                    pos = np.where(tmpc > tot*0.5);  tmpc[pos] -= tot
                    neg = np.where(tmpc < -tot*0.5); tmpc[neg] += tot
                    iqr  = np.percentile(tmpc[id],75) - np.percentile(tmpc[id],25)
                    outl = np.where(np.abs(tmpc) > iqr*1.5)
                    if len(outl[0]) > 0:
                        wjd[outl]=0.; wd[outl]=0.; wm[outl]=0.; wy[outl]=0.
                    onset_jday[:,it,jt]=wjd; onset_day[:,it,jt]=wd
                    onset_month[:,it,jt]=wm; onset_year[:,it,jt]=wy

                    # ONSET second pass
                    outl = np.where(wjd==0.)
                    if len(outl[0]) > 0:
                        wjd[:]=0.; wd[:]=0.; wm[:]=0.; wy[:]=0.
                        sid     = np.where(jday == sdate)[0]
                        sjday_  = jday[sid];  sday_   = day[sid]
                        smonth_ = month[sid]; syear_  = year[sid]
                        _wjd, _wd, _wm, _wy = rainyseason_B17_onset(
                            nyrs, tot_int, jday, day, month, year,
                            sdate, ap, npass, sjday_, sday_, smonth_, syear_)
                        n = min(len(_wjd), nyrs)
                        wjd[:n]=_wjd[:n]; wd[:n]=_wd[:n]; wm[:n]=_wm[:n]; wy[:n]=_wy[:n]
                        onset_jday[outl,it,jt]=wjd[outl]; onset_day[outl,it,jt]=wd[outl]
                        onset_month[outl,it,jt]=wm[outl]; onset_year[outl,it,jt]=wy[outl]

                        wjd[:] = onset_jday[:,it,jt]
                        miss = np.where(wjd==0.); id = np.where(wjd!=0.)
                        if len(id[0]) > 0:
                            tmpx = np.cos(wjd*math.pi/183.); tmpy = np.sin(wjd*math.pi/183.)
                            med  = math.atan2(np.median(tmpy[id]),np.median(tmpx[id]))*183./math.pi
                            if med < 0.: med += tot
                            tmpc = wjd[:] - med
                            if len(miss[0]) > 0: tmpc[miss] = 0.
                            pos = np.where(tmpc > tot*0.5);  tmpc[pos] -= tot
                            neg = np.where(tmpc < -tot*0.5); tmpc[neg] += tot
                            iqr  = np.percentile(tmpc[id],75) - np.percentile(tmpc[id],25)
                            outl = np.where(np.abs(tmpc) > iqr*3.)
                            if len(outl[0]) > 0:
                                onset_jday[outl,it,jt]=0.; onset_day[outl,it,jt]=0.
                                onset_month[outl,it,jt]=0.; onset_year[outl,it,jt]=0.

                # DEMISE first pass
                djd[:]=0.; dd[:]=0.; dm[:]=0.; dy[:]=0.; dsc[:]=0.
                djd[:],dd[:],dm[:],dy[:],dsc[:,:] = rainyseason_demise(
                    nyrs, tot_int, jday, day, month, year,
                    sdate, ap, djd, dd, dm, dy, dsc)
                miss = np.where(djd==0.); id = np.where(djd!=0.)
                if len(id[0]) > 0:
                    tmpx = np.cos(djd*math.pi/183.); tmpy = np.sin(djd*math.pi/183.)
                    med  = math.atan2(np.median(tmpy[id]),np.median(tmpx[id]))*183./math.pi
                    if med < 0.: med += tot
                    tmpc = djd[:] - med
                    if len(miss[0]) > 0: tmpc[miss] = 0.
                    pos = np.where(tmpc > tot*0.5);  tmpc[pos] -= tot
                    neg = np.where(tmpc < -tot*0.5); tmpc[neg] += tot
                    iqr  = np.percentile(tmpc[id],75) - np.percentile(tmpc[id],25)
                    outl = np.where(np.abs(tmpc) > iqr*1.5)
                    if len(outl[0]) > 0:
                        djd[outl]=0.; dd[outl]=0.; dm[outl]=0.; dy[outl]=0.
                    demise_jday[:,it,jt]=djd; demise_day[:,it,jt]=dd
                    demise_month[:,it,jt]=dm; demise_year[:,it,jt]=dy

                    # DEMISE second pass
                    outl = np.where(djd==0.)
                    if len(outl[0]) > 0:
                        djd[:]=0.; dd[:]=0.; dm[:]=0.; dy[:]=0.
                        sid     = np.where(jday == sdate)[0]
                        sjday_  = jday[sid];  sday_   = day[sid]
                        smonth_ = month[sid]; syear_  = year[sid]
                        _djd, _dd, _dm, _dy = rainyseason_B17_demise(
                            nyrs, tot_int, jday, day, month, year,
                            sdate, ap, npass, sjday_, sday_, smonth_, syear_)
                        n = min(len(_djd), nyrs)
                        djd[:n]=_djd[:n]; dd[:n]=_dd[:n]; dm[:n]=_dm[:n]; dy[:n]=_dy[:n]
                        demise_jday[outl,it,jt]=djd[outl]; demise_day[outl,it,jt]=dd[outl]
                        demise_month[outl,it,jt]=dm[outl]; demise_year[outl,it,jt]=dy[outl]

                        djd[:] = demise_jday[:,it,jt]
                        miss = np.where(djd==0.); id = np.where(djd!=0.)
                        if len(id[0]) > 0:
                            tmpx = np.cos(djd*math.pi/183.); tmpy = np.sin(djd*math.pi/183.)
                            med  = math.atan2(np.median(tmpy[id]),np.median(tmpx[id]))*183./math.pi
                            if med < 0.: med += tot
                            tmpc = djd[:] - med
                            if len(miss[0]) > 0: tmpc[miss] = 0.
                            pos = np.where(tmpc > tot*0.5);  tmpc[pos] -= tot
                            neg = np.where(tmpc < -tot*0.5); tmpc[neg] += tot
                            iqr  = np.percentile(tmpc[id],75) - np.percentile(tmpc[id],25)
                            outl = np.where(np.abs(tmpc) > iqr*3.)
                            if len(outl[0]) > 0:
                                demise_jday[outl,it,jt]=0.; demise_day[outl,it,jt]=0.
                                demise_month[outl,it,jt]=0.; demise_year[outl,it,jt]=0.

                # 33 % missing mask
                if len(np.where(onset_jday[:,it,jt]==0.)[0])/float(nyrs) > 0.33:
                    rm[it,jt]=0.
                    onset_jday[:,it,jt]=0.; onset_day[:,it,jt]=0.
                    onset_month[:,it,jt]=0.; onset_year[:,it,jt]=0.
                if len(np.where(demise_jday[:,it,jt]==0.)[0])/float(nyrs) > 0.33:
                    rm[it,jt]=0.
                    demise_jday[:,it,jt]=0.; demise_day[:,it,jt]=0.
                    demise_month[:,it,jt]=0.; demise_year[:,it,jt]=0.

                # Demise year rearrangement
                if demise_year[1,it,jt]==float(yr0) or demise_year[2,it,jt]==float(yr0+1):
                    demise_year[0:nyrs-1,it,jt]  = demise_year[1:nyrs,it,jt];  demise_year[nyrs-1,it,jt]=0.
                    demise_month[0:nyrs-1,it,jt] = demise_month[1:nyrs,it,jt]; demise_month[nyrs-1,it,jt]=0.
                    demise_day[0:nyrs-1,it,jt]   = demise_day[1:nyrs,it,jt];   demise_day[nyrs-1,it,jt]=0.
                    demise_jday[0:nyrs-1,it,jt]  = demise_jday[1:nyrs,it,jt];  demise_jday[nyrs-1,it,jt]=0.

    print("  Step 4 done.")

    # ── Step 5a: duration and accumulation ───────────────────────────────────

    durwet = np.zeros((nyrs, nlat, nlon))
    durdry = np.zeros((nyrs, nlat, nlon))
    totwet = np.zeros((nyrs, nlat, nlon))
    totdry = np.zeros((nyrs, nlat, nlon))

    for it in range(nlat):
        for jt in range(nlon):
            for yt in range(nyrs):
                if onset_year[yt,it,jt]==0. or demise_year[yt,it,jt]==0.:
                    continue

                # Number of days in the onset year (for correct index offset on leap years)
                oy = int(onset_year[yt,it,jt])
                dy_ = int(demise_year[yt,it,jt])

                # Index offset: count actual days from yr0 up to each year boundary.
                # This handles leap years correctly when mapping (year, DOY) → flat index.
                def year_offset(y):
                    return sum(
                        366 if pd.Timestamp(f"{yy}-01-01").is_leap_year else 365
                        for yy in range(yr0, y)
                    )

                if demise_year[yt,it,jt] == onset_year[yt,it,jt]:
                    if demise_jday[yt,it,jt] < onset_jday[yt,it,jt]:
                        beg = year_offset(oy)  + int(demise_jday[yt,it,jt]) - 1
                        ned = year_offset(oy)  + int(onset_jday[yt,it,jt])  - 1
                        if 0 <= beg < ned <= ntot:
                            durdry[yt,it,jt]=float(ned-beg); totdry[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                        if yt < nyrs-1 and demise_year[yt+1,it,jt] > 0.:
                            dy1 = int(demise_year[yt+1,it,jt])
                            beg = year_offset(oy)  + int(onset_jday[yt,it,jt])   - 1
                            ned = year_offset(dy1) + int(demise_jday[yt+1,it,jt]) - 1
                            if 0 <= beg < ned <= ntot:
                                durwet[yt,it,jt]=float(ned-beg); totwet[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                    elif onset_jday[yt,it,jt] < demise_jday[yt,it,jt]:
                        beg = year_offset(oy)  + int(onset_jday[yt,it,jt])  - 1
                        ned = year_offset(oy)  + int(demise_jday[yt,it,jt]) - 1
                        if 0 <= beg < ned <= ntot:
                            durwet[yt,it,jt]=float(ned-beg); totwet[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                        if yt < nyrs-1 and onset_year[yt+1,it,jt] > 0.:
                            oy1 = int(onset_year[yt+1,it,jt])
                            beg = year_offset(oy)  + int(demise_jday[yt,it,jt])  - 1
                            ned = year_offset(oy1) + int(onset_jday[yt+1,it,jt]) - 1
                            if 0 <= beg < ned <= ntot:
                                durdry[yt,it,jt]=float(ned-beg); totdry[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                elif 0. < demise_year[yt,it,jt] < onset_year[yt,it,jt]:
                    beg = year_offset(dy_) + int(demise_jday[yt,it,jt]) - 1
                    ned = year_offset(oy)  + int(onset_jday[yt,it,jt])  - 1
                    if 0 <= beg < ned <= ntot:
                        durdry[yt,it,jt]=float(ned-beg); totdry[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                    if yt < nyrs-1 and demise_year[yt+1,it,jt] > 0.:
                        if onset_jday[yt,it,jt] < demise_jday[yt+1,it,jt]:
                            dy1 = int(demise_year[yt+1,it,jt])
                            beg = year_offset(oy)  + int(onset_jday[yt,it,jt])   - 1
                            ned = year_offset(dy1) + int(demise_jday[yt+1,it,jt]) - 1
                            if 0 <= beg < ned <= ntot:
                                durwet[yt,it,jt]=float(ned-beg); totwet[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                elif 0. < onset_year[yt,it,jt] < demise_year[yt,it,jt]:
                    beg = year_offset(oy)  + int(onset_jday[yt,it,jt])  - 1
                    ned = year_offset(dy_) + int(demise_jday[yt,it,jt]) - 1
                    if 0 <= beg < ned <= ntot:
                        durwet[yt,it,jt]=float(ned-beg); totwet[yt,it,jt]=np.sum(prec[beg:ned,it,jt])
                    if yt < nyrs-1 and onset_year[yt+1,it,jt] > 0.:
                        if demise_jday[yt,it,jt] < onset_jday[yt+1,it,jt]:
                            oy1 = int(onset_year[yt+1,it,jt])
                            beg = year_offset(dy_) + int(demise_jday[yt,it,jt])  - 1
                            ned = year_offset(oy1) + int(onset_jday[yt+1,it,jt]) - 1
                            if 0 <= beg < ned <= ntot:
                                durdry[yt,it,jt]=float(ned-beg); totdry[yt,it,jt]=np.sum(prec[beg:ned,it,jt])

    # Cap impossible durations
    for arr in [durwet, durdry]:
        arr[arr > tot_int] = 0.
    totwet[durwet == 0.] = 0.
    totdry[durdry == 0.] = 0.

    print("  Step 5a done.")

    # ── Step 5b: zero → missval ───────────────────────────────────────────────

    yrs_coord = np.arange(yr0, yr0 + nyrs, 1)
    nyrs_save = nyrs - 2
    yr_start_coord = str(yrs_coord[0])
    yr_end_coord   = str(yrs_coord[nyrs_save - 1])

    for arr in [onset_jday, onset_day, onset_month, onset_year,
                demise_jday, demise_day, demise_month, demise_year,
                totwet, totdry, durwet, durdry]:
        arr[arr == 0.] = missval

    # ── Step 5c: save ─────────────────────────────────────────────────────────

    coords = {
        "year": yrs_coord[:nyrs_save],
        "doy":  np.arange(1, tot_int + 1),
        "lat":  lats,
        "lon":  lons,
    }

    ds_out = xr.Dataset(
        {
            "onset_jday":   make_var(onset_jday,   "Wet season onset (Day of Year)",        "day_of_year", nyrs_save),
            "onset_day":    make_var(onset_day,    "Wet season onset (day of month)",        "1",           nyrs_save),
            "onset_month":  make_var(onset_month,  "Wet season onset (month)",               "1",           nyrs_save),
            "onset_year":   make_var(onset_year,   "Wet season onset (year)",                "1",           nyrs_save),
            "demise_jday":  make_var(demise_jday,  "Wet season demise (Day of Year)",        "day_of_year", nyrs_save),
            "demise_day":   make_var(demise_day,   "Wet season demise (day of month)",       "1",           nyrs_save),
            "demise_month": make_var(demise_month, "Wet season demise (month)",              "1",           nyrs_save),
            "demise_year":  make_var(demise_year,  "Wet season demise (year)",               "1",           nyrs_save),
            "totwet":       make_var(totwet,       "Accumulated precipitation (wet season)", "mm",          nyrs_save),
            "totdry":       make_var(totdry,       "Accumulated precipitation (dry season)", "mm",          nyrs_save),
            "durwet":       make_var(durwet,       "Duration of the wet season",             "days",        nyrs_save),
            "durdry":       make_var(durdry,       "Duration of the dry season",             "days",        nyrs_save),
            "smoothed_cycle": xr.DataArray(
                smoothed,
                dims=["doy", "lat", "lon"],
                attrs={"long_name": "Smoothed mean annual cycle (3 harmonics)", "units": "mm/day"},
            ),
        },
        coords=coords,
        attrs={
            "dataset":  "TRMM 3B42 v7",
            "period":   f"{yr_start_coord}-{yr_end_coord}",
            "method":   "Bombardi et al. (2019) doi:10.1175/BAMS-D-18-0177.1",
            "note":     "Leap-day (DOY 366) mapped to DOY 365 for harmonic cycle.",
        },
    )

    encoding = {v: {"zlib": True, "complevel": 4, "_FillValue": missval}
                for v in ds_out.data_vars}
    ds_out.to_netcdf(outfile, encoding=encoding)
    print(f"  Saved → {outfile}")
    ds_out.close()

print("\nAll done.")

