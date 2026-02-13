import numpy as np
import xarray as xr
import myfunctions as mf

def reorder_south2north(data, lat):
    # if latitude is not indexed from SP to NP, then reorder
    if lat[0]>lat[1]:
        lat = lat[::-1]
        data  = data[::-1]
    return data, lat

def get_itczposition_adam(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    return np.nansum(lati * areai * pri) / np.nansum(areai * pri)

def test_itczposition_adam(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    # calculate itcz position according to Adam
    itcz = get_itczposition_adam(pr, lat, latboundary, dlat)
    # lat index corresponding to itcz 
    ilati = np.argmin(np.abs(itcz - lati))
    aux1=np.abs(np.nansum((lati[0:ilati+1]-itcz)*pri[0:ilati+1]*areai[0:ilati+1]))
    aux2=np.abs(np.nansum((lati[ilati+1:]-itcz)*pri[ilati+1:]*areai[ilati+1:]))
    return aux1, aux2    
    
def get_itczposition_voigt(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    # area-integrated precip (up to constant factor)
    tot = np.sum(pri*areai)
    # integrated pri from southern latboundary to lati
    pri_int = np.zeros(lati.size) + np.nan
    for j in range(0, lati.size):
        pri_int[j] = np.sum(pri[0:j+1]*areai[0:j+1])
    # itcz is where integrated pri is 0.5 of total area-integrated pri
    return lati[np.argmin(np.abs(pri_int - 0.5*tot))]    

def test_itczposition_voigt(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    # calculate itcz position according to Adam
    itcz = get_itczposition_voigt(pr, lat, latboundary, dlat)
    # lat index corresponding to itcz 
    ilati = np.argmin(np.abs(itcz - lati))
    aux1=np.nansum(pri[0:ilati+1]*areai[0:ilati+1])
    aux2=np.nansum(pri[ilati+1:]*areai[ilati+1:])
    return aux1, aux2

def itcz_adam_lonwise(da_ann, lat, londim, year_slice, latboundary=20, dlat=0.1):
    """
    da_ann: DataArray with dims (year, lat, lon)
    returns: DataArray with dims (year, lon)
    """
    results = []

    for yr in da_ann.sel(year=year_slice).year.values:
        da_yr = da_ann.sel(year=yr)

        # loop over longitude (still unavoidable, but only 1D slices)
        itcz_lon = []
        for lo in da_yr[londim].values:
            pr_merid = da_yr.sel({londim: lo})

            itcz = get_itczposition_adam(
                pr_merid,
                lat,
                latboundary=latboundary,
                dlat=dlat,
            )
            itcz_lon.append(itcz)

        itcz_lon = xr.DataArray(
            itcz_lon,
            dims=[londim],
            coords={londim: da_yr[londim]},
        )

        results.append(itcz_lon.assign_coords(year=yr))

    return xr.concat(results, dim="year")


def compute_adam_ITCZ(
    model_name,
    model_meta,
    experiments,
    base_dir,
    lon_slice,
    season,
    time_slice,
    latboundary=20,
    dlat=0.1,
):
    """
    Compute Adam et al. (2016) ITCZ latitude (lon-wise).

    Returns
    -------
    dict
        {experiment: DataArray(year, lon)}
    """

    itcz_by_exp = {}

    for exp, meta in experiments.items():

        # --- skip historical if present ---
        if meta["project"] == "CMIP":
            continue

        # --- ensemble handling (copied from your code) ---
        if model_name == "CESM2-WACCM":
            if meta["scenario"] == "G6sulfur":
                ensemble = "r1i1p1f2"
            else:
                ensemble = "r1i1p1f1"
        else:
            ensemble = model_meta["ensemble"]

        # --- build CEDA path ---
        base = (
            base_dir
            / meta["project"]
            / model_meta["institution"]
            / model_name
            / meta["scenario"]
            / ensemble
            / "Amon"
            / "pr"
            / model_meta["grid"]
            / "latest"
        )

        # --- open dataset (with special cases) ---
        if model_name == "CESM2-WACCM":
            if meta["scenario"] == "G6sulfur":
                ds = mf.open_files_CESM_G6sulfur(base)
            elif meta["scenario"] == "ssp585":
                ds = mf.open_files_CESM_ssp585(base)
            else:
                ds = mf.open_files(str(base))
        elif model_name == "IPSL-CM6A-LR":
            if meta["scenario"] == "ssp585":
                ds = mf.open_files_IPSL_ssp585(base)
            else:
                ds = mf.open_files(str(base))
        else:
            ds = mf.open_files(str(base))

        # --- read precipitation ---
        pr = mf.read_var(ds, "pr")

        # --- longitude slice ---
        if lon_slice is not None:
            pr = pr.sel(lon=lon_slice)

        # --- seasonal / annual mean by year ---
        start_month, end_month = season
        pr_ann = mf.seasonal_mean_by_year(pr, start_month, end_month)

        # --- longitude dimension ---
        londim = mf.get_lon_dim(pr_ann)

        # --- Adam ITCZ ---
        itcz_by_exp[exp] = itcz_adam_lonwise(
            da_ann=pr_ann,
            lat=pr_ann.lat,
            londim=londim,
            year_slice=time_slice,
            latboundary=latboundary,
            dlat=dlat,
        )

    return itcz_by_exp

