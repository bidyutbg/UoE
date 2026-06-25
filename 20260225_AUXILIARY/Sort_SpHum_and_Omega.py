#!/usr/bin/env python
# coding: utf-8

# In[1]:


# sort q profiles with respect to w500 and plot, and sort w profiles with respect to q750.


# # Load Data

# In[2]:


import xarray as xr
import numpy as np
import intake
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["online"]


# In[3]:


#In order to open a dataset from the catalog below, use the catalog as follows:
ds = cat["ERA5"](zoom=7).to_dask()


# In[4]:


q=ds.q
w=ds.w


# In[5]:


print(q)
print(w)


# # Example year of 2020

# In[6]:


def _format_level(base_level) -> str:
    """Return a compact string for scalar or list of levels."""
    levels = np.atleast_1d(base_level)
    if len(levels) == 1:
        return f"{int(levels[0])}"
    else:
        return f"{int(levels.min())}–{int(levels.max())}"  # e.g. "800–1000"

def sort_profiles_by_level(
    da_base: xr.DataArray,
    da_target: xr.DataArray,
    base_level,
    percentiles=None,
    time_slice=None,
    season: str | None = None,
    lat_bounds: tuple | None = None,   # e.g. (-30, 30) for tropics
    output_file: str | None = None,
) -> xr.DataArray:
    import os

    # ── early exit if file already exists ─────────────────────────────────────
    if output_file is not None and os.path.exists(output_file):
        print(f"File exists, loading: {output_file}")
        return xr.open_dataarray(output_file)

    if percentiles is None:
        percentiles = np.arange(0, 101, 1)

    # ── time selection ────────────────────────────────────────────────────────
    # if time_slice is not None:
    #     da_base   = da_base.sel(time=time_slice)
    #     da_target = da_target.sel(time=time_slice)
    # ── time selection ────────────────────────────────────────────────────────
    if time_slice is not None:
        if isinstance(time_slice, slice):
            da_base   = da_base.sel(time=time_slice)
            da_target = da_target.sel(time=time_slice)
        else:
            # assume boolean array/DataArray indexer (e.g. El Niño mask)
            time_slice = np.asarray(time_slice)   # ensure numpy, not dask
            da_base   = da_base.isel(time=time_slice)
            da_target = da_target.isel(time=time_slice)

    if season is not None:
        season_months = {
            "DJF": [12, 1, 2], "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],  "SON": [9, 10, 11],
        }
        months = season_months[season.upper()]
        mask = da_base.time.dt.month.isin(months)
        da_base   = da_base.isel(time=mask)
        da_target = da_target.isel(time=mask)

    # ── spatial crop by latitude ──────────────────────────────────────────────
    if lat_bounds is not None:
        lat_min, lat_max = lat_bounds
        lat = da_base.lat.compute()           # force lat to numpy
        cell_mask = ((lat >= lat_min) & (lat <= lat_max)).compute()  # force mask to numpy
        da_base   = da_base.isel(cell=cell_mask)
        da_target = da_target.isel(cell=cell_mask)
        print(f"  Cropped to lat [{lat_min}, {lat_max}]: "
              f"{int(cell_mask.sum())} / {len(cell_mask)} cells retained")

    # ── extract / average the sorting key at base_level ──────────────────────
    levels = np.atleast_1d(base_level)
    if len(levels) == 1:
        key = da_base.sel(level=levels[0], method="nearest")
    else:
        key = da_base.sel(level=levels).mean("level")

    # ── compute sorted profiles time-step by time-step ───────────────────────
    n_time   = da_target.sizes["time"]
    n_levels = da_target.sizes["level"]
    n_pct    = len(percentiles)

    result = np.full((n_time, n_levels, n_pct), np.nan, dtype=np.float32)

    for t in range(n_time):
        key_t    = key.isel(time=t).values
        target_t = da_target.isel(time=t).values

        bin_edges = np.nanpercentile(key_t, percentiles)

        for p_idx in range(n_pct - 1):
            lo, hi = bin_edges[p_idx], bin_edges[p_idx + 1]
            mask = (key_t >= lo) & (key_t < hi)
            if mask.sum() > 0:
                result[t, :, p_idx] = target_t[:, mask].mean(axis=-1)

        mask = key_t >= bin_edges[-1]
        if mask.sum() > 0:
            result[t, :, -1] = target_t[:, mask].mean(axis=-1)

    # ── wrap in DataArray ─────────────────────────────────────────────────────
    out = xr.DataArray(
        result,
        dims=["time", "level", "percentile"],
        coords={
            "time":       da_target.time.values,
            "level":      da_target.level.values,
            "percentile": percentiles,
        },
        name=da_target.name,
        attrs={
            **da_target.attrs,
            "sorting_variable": da_base.name,
            "sorting_level":    _format_level(np.atleast_1d(base_level)),  # ← clean string
            "lat_bounds":       str(lat_bounds),
        },
    )

    if output_file is not None:
        print(f"Saving → {output_file}")
        out.to_netcdf(output_file)

    return out


# In[7]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── select 2020 ───────────────────────────────────────────────────────────────
t2020 = slice("2020-01", "2020-12")

# Sort q profiles by w at 500 hPa
q_sorted_by_w500_2020 = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    percentiles= np.arange(0, 101, 1),
    time_slice = t2020,
    output_file= "q_sorted_by_w500_2020.nc",
)

# Sort q profiles by w at 500 hPa : tropics only
q_sorted_by_w500_2020_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = slice("2020-01", "2020-12"),
    lat_bounds = (-30, 30),          # ← tropics
    output_file= "q_sorted_by_w500_2020_tropics.nc",
)

# # Sort w profiles by q at 750 hPa
# w_sorted = sort_profiles_by_level(
#     da_base    = q,
#     da_target  = w,
#     base_level = 750,
#     percentiles= np.arange(0, 101, 1),
#     time_slice = t2020,
#     output_file= "w_sorted_by_q750_2020.nc",
# )

# Sort w profiles by q at 800-1000 hPa
w_sorted_by_qLT_2020 = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800.,  825., 850.,  875.,  900.,  925.,  950.,  975., 1000.],
    percentiles= np.arange(0, 101, 1),
    time_slice = t2020,
    output_file= "w_sorted_by_qLT_2020.nc",
)

# tropics only
w_sorted_by_qLT_2020_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800.,  825., 850.,  875.,  900.,  925.,  950.,  975., 1000.],
    time_slice = slice("2020-01", "2020-12"),
    lat_bounds = (-30, 30),          # ← tropics
    output_file= "w_sorted_by_qLT_2020_tropics.nc",
)


# In[ ]:





# In[8]:


# # ── plotting helper ───────────────────────────────────────────────────────────
# def plot_sorted_profiles(da_sorted, ax, title, xlabel, cmap="RdBu_r", 
#                          center=None, vmin=None, vmax=None):
#     """
#     da_sorted: DataArray (time, level, percentile)
#     Plots a hovmöller-style (percentile × level) plot, one panel per month.
#     Or time-mean if you want a quick overview.
#     """
#     # time-mean for a compact overview  (no time averaging in the sort — just display)
#     data = da_sorted.mean("time")   # (level, percentile)

#     levels = da_sorted.level.values
#     pcts   = da_sorted.percentile.values

#     if center is not None:
#         vmax = vmax or np.nanpercentile(np.abs(data.values), 98)
#         norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=-vmax, vmax=vmax)
#     else:
#         norm = None

#     pcm = ax.pcolormesh(pcts, levels, data.values,
#                         cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
#                         shading="auto")
#     ax.invert_yaxis()
#     ax.set_xlabel("Percentile of sorting variable", fontsize=11)
#     ax.set_ylabel("Pressure (hPa)", fontsize=11)
#     ax.set_title(title, fontsize=12, fontweight="bold")
#     return pcm

# # Version#2
# def plot_sorted_profiles(da_sorted, ax, title, xlabel, cmap="RdBu_r", 
#                          center=None, vmin=None, vmax=None):
#     data = da_sorted.mean("time")   # (level, percentile)

#     levels = da_sorted.level.values
#     pcts   = da_sorted.percentile.values

#     if center is not None:
#         vmax = vmax or np.nanpercentile(np.abs(data.values), 98)
#         norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=-vmax, vmax=vmax)
#         pcm = ax.pcolormesh(pcts, levels, data.values,
#                             cmap=cmap, norm=norm,        # ← no vmin/vmax here
#                             shading="auto")
#     else:
#         pcm = ax.pcolormesh(pcts, levels, data.values,
#                             cmap=cmap, vmin=vmin, vmax=vmax,
#                             shading="auto")

#     ax.invert_yaxis()
#     ax.set_xlabel("Percentile of sorting variable", fontsize=11)
#     ax.set_ylabel("Pressure (hPa)", fontsize=11)
#     ax.set_title(title, fontsize=12, fontweight="bold")
#     return pcm

def plot_sorted_profiles(da_sorted, ax, title, xlabel, cmap="RdBu_r", 
                         center=None, vmin=None, vmax=None):
    data = da_sorted.mean("time")   # (level, percentile)
    levels = da_sorted.level.values
    pcts   = da_sorted.percentile.values

    # ── infer sorting variable from attrs ────────────────────────────────────
    sort_var  = da_sorted.attrs.get("sorting_variable", "")
    sort_lev  = da_sorted.attrs.get("sorting_level", "")

    # x-axis label: "Percentile of w₅₀₀ (Pa s⁻¹)" etc.
    var_labels = {"w": f"w$_{{{sort_lev}}}$ (Pa s⁻¹)",
                  "q": f"q$_{{{sort_lev}}}$ (kg kg⁻¹)"}
    xlabel = f"Percentile of {var_labels.get(sort_var, sort_var)}"

    # low/high end annotations
    if sort_var == "w":        # w: large positive = descent, large negative = ascent
        lo_label, hi_label = "Ascent\n(−w)", "Descent\n(+w)"
    elif sort_var == "q":      # q: low = dry, high = moist
        lo_label, hi_label = "Dry\n(low q)", "Moist\n(high q)"
    else:
        lo_label, hi_label = "Low", "High"

    if center is not None:
        vmax = vmax or np.nanpercentile(np.abs(data.values), 98)
        norm = mcolors.TwoSlopeNorm(vcenter=center, vmin=-vmax, vmax=vmax)
        pcm = ax.pcolormesh(pcts, levels, data.values,
                            cmap=cmap, norm=norm,
                            shading="auto")
    else:
        pcm = ax.pcolormesh(pcts, levels, data.values,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="auto")

    # Zero contour (bold)
    ax.contour(pcts, levels, data.values, levels=[0], colors='black', linewidths=1.5)
    ax.contour(pcts, levels, data.values, levels=[-0.04], colors='blue', linewidths=2.5)
    print(xlabel)
    if xlabel == "Percentile of w$_{500}$ (Pa s⁻¹)":
        ax.contour(pcts, levels, data.values, levels=[0.006], colors='black', linewidths=1.5)



    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # annotate low/high ends just inside the axes
    ax_kw = dict(transform=ax.transAxes, fontsize=9, va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
    ax.text(0.02, 0.5, lo_label, ha="left",  **ax_kw)
    ax.text(0.98, 0.5, hi_label, ha="right", **ax_kw)

    return pcm


# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: q sorted by w500
pcm1 = plot_sorted_profiles(
    q_sorted_by_w500_2020, axes[0],
    title  = "q sorted by w$_{500}$ — 2020",
    xlabel = "Percentile of w500",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0], label="q (kg kg⁻¹)")

# Panel 2: w sorted by q750
pcm2 = plot_sorted_profiles(
    w_sorted_by_qLT_2020, axes[1],
    title  = "w sorted by qLT — 2020",
    xlabel = "Percentile of qLT",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[1], label="w (Pa s⁻¹)")

plt.suptitle("Sorted vertical profiles — ERA5 hackathon catalog (2020)", 
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("sorted_profiles_2020.png", dpi=150, bbox_inches="tight")
plt.show()


# In[10]:


def plot_sorting_key(da_base, ax, base_level, time_slice=None, 
                     color="steelblue", label=None):
    """
    Plot the mean (across time) of the sorting key variable at base_level,
    binned by percentile — i.e. the x-axis variable of the sorted profiles.
    """
    if time_slice is not None:
        da_base = da_base.sel(time=time_slice)

    # extract level
    key = da_base.sel(level=base_level, method="nearest")  # (time, cell)

    # for each timestep, compute percentile values, then average across time
    percentiles = np.arange(0, 101, 1)
    pct_values  = np.nanpercentile(key.values, percentiles, axis=-1)  # (101, time)
    pct_mean    = pct_values.mean(axis=-1)   # (101,)
    pct_std     = pct_values.std(axis=-1)    # (101,) — spread across months

    var_name  = da_base.name
    units     = da_base.attrs.get("units", "")
    label     = label or f"{var_name}$_{{{base_level}}}$"

    ax.plot(percentiles, pct_mean, color=color, lw=2, label=label)
    # ax.fill_between(percentiles, pct_mean - pct_std, pct_mean + pct_std,
                    # alpha=0.25, color=color, label="±1σ across months")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(50, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Percentile", fontsize=11)
    ax.set_ylabel(f"{label} ({units})", fontsize=11)
    ax.set_title(f"Sorting key: {label} by percentile — 2020", 
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)


# In[11]:


# ── plot ──────────────────────────────────────────────────────────────────────
t2020 = slice("2020-01", "2020-12")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

plot_sorting_key(w, axes[0], base_level=500, time_slice=t2020, color="steelblue")
plot_sorting_key(q, axes[1], base_level=850, time_slice=t2020, color="darkorange")

plt.suptitle("Sorting key distributions — ERA5 2020", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("sorting_keys_2020.png", dpi=150, bbox_inches="tight")
plt.show()


# In[12]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],   # top row twice the height of bottom row
    hspace=0.15,            # vertical space between rows
    wspace=0.25,            # horizontal space between columns
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── Row 0: sorted profiles ────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    q_sorted_by_w500_2020, axes[0, 0],
    title  = "q sorted by w$_{500}$ — 2020",
    xlabel = "",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0, 0], label="q (kg kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    w_sorted_by_qLT_2020, axes[0, 1],
    title  = "w sorted by qLT — 2020",
    xlabel = "",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="w (Pa s⁻¹)", pad=0.02)

# remove x tick labels on top row so bottom row x-axis is the reference
for ax in axes[0]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# ── Row 1: sorting keys ───────────────────────────────────────────────────────
plot_sorting_key(w, axes[1, 0], base_level=500, time_slice=t2020, color="steelblue")
plot_sorting_key(q, axes[1, 1], base_level=850, time_slice=t2020, color="darkorange")

# enforce identical x limits across all columns so percentiles align
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    axes[1, col].set_xlim(0, 100)

plt.suptitle("Sorted vertical profiles and sorting key distributions — ERA5 2020",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

# ── force same left/right edges for both columns so widths truly match ────────
# (colorbar on top row shifts its axes; this corrects for that)
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0,           # align left edge
        pos_bot.y0,           # keep vertical position
        pos_top.width,        # match width exactly
        pos_bot.height,       # keep its own height
    ])

plt.savefig("4panel_sorted_profiles_2020.png", dpi=150, bbox_inches="tight")
plt.show()


# In[13]:


def crop_by_latitude(lat_bounds, *data_arrays):
    """
    Crop one or more xarray DataArrays by latitude bounds.

    Parameters
    ----------
    lat_bounds : tuple[float, float] | None
        (lat_min, lat_max) to retain, or None to skip cropping.
    *data_arrays : xr.DataArray
        Any number of DataArrays sharing the same 'cell' and 'lat' dimensions.

    Returns
    -------
    list[xr.DataArray]
        Cropped DataArrays in the same order as provided.
        If lat_bounds is None, returns the originals unchanged.
    """
    if lat_bounds is None:
        return list(data_arrays)

    lat_min, lat_max = lat_bounds

    # Use the lat coordinate from the first array (assumed shared across all)
    lat = data_arrays[0].lat.compute()
    cell_mask = ((lat >= lat_min) & (lat <= lat_max)).compute()

    cropped = [da.isel(cell=cell_mask) for da in data_arrays]

    print(
        f"  Cropped to lat [{lat_min}, {lat_max}]: "
        f"{int(cell_mask.sum())} / {len(cell_mask)} cells retained"
    )

    return cropped


# In[14]:


lat_bounds=(-30,30)
w_trop, q_trop = crop_by_latitude(lat_bounds, w, q)


# In[15]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],   # top row twice the height of bottom row
    hspace=0.15,            # vertical space between rows
    wspace=0.25,            # horizontal space between columns
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── Row 0: sorted profiles ────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    q_sorted_by_w500_2020_tropics, axes[0, 0],
    title  = "q sorted by w$_{500}$ — 2020",
    xlabel = "",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0, 0], label="q (kg kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    w_sorted_by_qLT_2020_tropics, axes[0, 1],
    title  = "w sorted by qLT — 2020",
    xlabel = "",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="w (Pa s⁻¹)", pad=0.02)

# remove x tick labels on top row so bottom row x-axis is the reference
for ax in axes[0]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# ── Row 1: sorting keys ───────────────────────────────────────────────────────
plot_sorting_key(w_trop, axes[1, 0], base_level=500, time_slice=t2020, color="steelblue")
plot_sorting_key(q_trop, axes[1, 1], base_level=850, time_slice=t2020, color="darkorange")

# enforce identical x limits across all columns so percentiles align
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    axes[1, col].set_xlim(0, 100)

plt.suptitle("Sorted vertical profiles and sorting key distributions — ERA5 2020 : Tropics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

# ── force same left/right edges for both columns so widths truly match ────────
# (colorbar on top row shifts its axes; this corrects for that)
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0,           # align left edge
        pos_bot.y0,           # keep vertical position
        pos_top.width,        # match width exactly
        pos_bot.height,       # keep its own height
    ])

plt.savefig("4panel_sorted_profiles_2020_tropics.png", dpi=150, bbox_inches="tight")
plt.show()


# # Warm El Nino Year 2016

# In[16]:


ElNinoYrs=[1983,1987,1988,1992,1995,1998,2003,2007,2010,2016]


# In[17]:


ElNinoYrs=[1983,1987,1988,1992,1995,1998,2003,2007,2010,2016]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── select 2020 ───────────────────────────────────────────────────────────────
t2016 = slice("2016-01", "2016-12")

# Sort q profiles by w at 500 hPa : tropics only
q_sorted_by_w500_2016_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = t2016,
    lat_bounds = (-30, 30),          # ← tropics
    output_file= "q_sorted_by_w500_2016_tropics_ElNino.nc",
)

# tropics only
w_sorted_by_qLT_2016_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800.,  825., 850.,  875.,  900.,  925.,  950.,  975., 1000.],
    time_slice = t2016,
    lat_bounds = (-30, 30),          # ← tropics
    output_file= "w_sorted_by_qLT_2016_tropics_ElNino.nc",
)


# In[18]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],   # top row twice the height of bottom row
    hspace=0.15,            # vertical space between rows
    wspace=0.25,            # horizontal space between columns
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── Row 0: sorted profiles ────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    q_sorted_by_w500_2016_tropics, axes[0, 0],
    title  = "q sorted by w$_{500}$ — 2016",
    xlabel = "",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0, 0], label="q (kg kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    w_sorted_by_qLT_2016_tropics, axes[0, 1],
    title  = "w sorted by qLT — 2016",
    xlabel = "",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="w (Pa s⁻¹)", pad=0.02)

# remove x tick labels on top row so bottom row x-axis is the reference
for ax in axes[0]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# ── Row 1: sorting keys ───────────────────────────────────────────────────────
plot_sorting_key(w_trop, axes[1, 0], base_level=500, time_slice=t2016, color="steelblue")
plot_sorting_key(q_trop, axes[1, 1], base_level=850, time_slice=t2016, color="darkorange")

# enforce identical x limits across all columns so percentiles align
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    axes[1, col].set_xlim(0, 100)

plt.suptitle("Sorted vertical profiles and sorting key distributions — ERA5 2016 (El Nino) : Tropics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

# ── force same left/right edges for both columns so widths truly match ────────
# (colorbar on top row shifts its axes; this corrects for that)
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0,           # align left edge
        pos_bot.y0,           # keep vertical position
        pos_top.width,        # match width exactly
        pos_bot.height,       # keep its own height
    ])

plt.savefig("4panel_sorted_profiles_2016_tropics_ElNino.png", dpi=150, bbox_inches="tight")
plt.show()


# # Cold La Nina Year 2008

# In[19]:


LaNinaYrs=[1974,1976,1989,1999,2000,2008,2011,2012,2021,2022]


# In[20]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── select 2020 ───────────────────────────────────────────────────────────────
t2008 = slice("2008-01", "2008-12")

# Sort q profiles by w at 500 hPa : tropics only
q_sorted_by_w500_2008_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = t2008,
    lat_bounds = (-30, 30),          # ← tropics
    output_file= "q_sorted_by_w500_2008_tropics_ElNino.nc",
)

# tropics only
w_sorted_by_qLT_2008_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800.,  825., 850.,  875.,  900.,  925.,  950.,  975., 1000.],
    time_slice = t2008,
    lat_bounds = (-30, 30),          # ← tropics
    output_file= "w_sorted_by_qLT_2008_tropics_ElNino.nc",
)


# In[21]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],   # top row twice the height of bottom row
    hspace=0.15,            # vertical space between rows
    wspace=0.25,            # horizontal space between columns
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── Row 0: sorted profiles ────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    q_sorted_by_w500_2008_tropics, axes[0, 0],
    title  = "q sorted by w$_{500}$ — 2008",
    xlabel = "",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0, 0], label="q (kg kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    w_sorted_by_qLT_2008_tropics, axes[0, 1],
    title  = "w sorted by qLT — 2008",
    xlabel = "",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="w (Pa s⁻¹)", pad=0.02)

# remove x tick labels on top row so bottom row x-axis is the reference
for ax in axes[0]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# ── Row 1: sorting keys ───────────────────────────────────────────────────────
plot_sorting_key(w_trop, axes[1, 0], base_level=500, time_slice=t2008, color="steelblue")
plot_sorting_key(q_trop, axes[1, 1], base_level=850, time_slice=t2008, color="darkorange")

# enforce identical x limits across all columns so percentiles align
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    axes[1, col].set_xlim(0, 100)

plt.suptitle("Sorted vertical profiles and sorting key distributions — ERA5 2008 (La Nina) : Tropics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

# ── force same left/right edges for both columns so widths truly match ────────
# (colorbar on top row shifts its axes; this corrects for that)
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0,           # align left edge
        pos_bot.y0,           # keep vertical position
        pos_top.width,        # match width exactly
        pos_bot.height,       # keep its own height
    ])

plt.savefig("4panel_sorted_profiles_2008_tropics_LaNina.png", dpi=150, bbox_inches="tight")
plt.show()


# # El Nino minus La Nina

# In[22]:


def plot_4panel_sorted_profiles(
    q_sorted_by_w500, w_sorted_by_qLT,
    w_trop, q_trop,
    time_slice,
    suptitle, savepath,
    q_label="q (kg kg⁻¹)", w_label="w (Pa s⁻¹)",
):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2, 1],
        hspace=0.15,
        wspace=0.25,
    )
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])

    # ── Row 0: sorted profiles ────────────────────────────────────────────────
    pcm1 = plot_sorted_profiles(
        q_sorted_by_w500, axes[0, 0],
        title  = f"q sorted by w$_{{500}}$",
        xlabel = "",
        cmap   = "RdBu_r",
    )
    fig.colorbar(pcm1, ax=axes[0, 0], label=q_label, pad=0.02)

    pcm2 = plot_sorted_profiles(
        w_sorted_by_qLT, axes[0, 1],
        title  = f"w sorted by qLT",
        xlabel = "",
        cmap   = "RdBu_r",
        center = 0,
    )
    fig.colorbar(pcm2, ax=axes[0, 1], label=w_label, pad=0.02)

    for ax in axes[0]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    # ── Row 1: sorting keys ───────────────────────────────────────────────────
    plot_sorting_key(w_trop, axes[1, 0], base_level=500, time_slice=time_slice, color="steelblue")
    plot_sorting_key(q_trop, axes[1, 1], base_level=850, time_slice=time_slice, color="darkorange")

    for col in range(2):
        axes[0, col].set_xlim(0, 100)
        axes[1, col].set_xlim(0, 100)

    plt.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    for col in range(2):
        pos_top = axes[0, col].get_position()
        pos_bot = axes[1, col].get_position()
        axes[1, col].set_position([
            pos_top.x0,
            pos_bot.y0,
            pos_top.width,
            pos_bot.height,
        ])

    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_4panel_difference(
    q_sorted_by_w500_elnino, q_sorted_by_w500_lanina,
    w_sorted_by_qLT_elnino,  w_sorted_by_qLT_lanina,
    w_trop_elnino, w_trop_lanina,
    q_trop_elnino, q_trop_lanina,
    t_elnino, t_lanina,
    suptitle="Sorted vertical profiles — El Niño minus La Niña : Tropics",
    savepath="4panel_difference_ElNino_minus_LaNina_tropics.png",
):
    # ── compute differences ───────────────────────────────────────────────────
    q_diff = q_sorted_by_w500_elnino.mean("time") - q_sorted_by_w500_lanina.mean("time")
    w_diff = w_sorted_by_qLT_elnino.mean("time")  - w_sorted_by_qLT_lanina.mean("time")
    # q_diff = q_sorted_by_w500_2016_tropics.mean("time") - q_sorted_by_w500_2008_tropics.mean("time")
    # w_diff = w_sorted_by_qLT_2016_tropics.mean("time")  - w_sorted_by_qLT_2008_tropics.mean("time")
    # # plot_sorted_profiles expects a 'time' dim — add a dummy one
    q_diff = q_diff.expand_dims("time")
    w_diff = w_diff.expand_dims("time")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2, 1],
        hspace=0.15,
        wspace=0.25,
    )
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])

    # ── Row 0: difference profiles ────────────────────────────────────────────
    # Symmetric colour limits so zero is white
    qlim = float(np.nanpercentile(np.abs(q_diff), 98))
    wlim = float(np.nanpercentile(np.abs(w_diff), 98))

    pcm1 = plot_sorted_profiles(
        q_diff, axes[0, 0],
        title  = "Δq sorted by w$_{500}$  (El Niño − La Niña)",
        xlabel = "",
        cmap   = "RdBu_r",
        center = 0,
        vmin=-qlim, vmax=qlim,          # pass through if your function accepts them
    )
    fig.colorbar(pcm1, ax=axes[0, 0], label="Δq (kg kg⁻¹)", pad=0.02)

    pcm2 = plot_sorted_profiles(
        w_diff, axes[0, 1],
        title  = "Δw sorted by qLT  (El Niño − La Niña)",
        xlabel = "",
        cmap   = "RdBu_r",
        center = 0,
        vmin=-wlim, vmax=wlim,
    )
    fig.colorbar(pcm2, ax=axes[0, 1], label="Δw (Pa s⁻¹)", pad=0.02)

    for ax in axes[0]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    # ── Row 1: both sorting-key distributions overlaid ────────────────────────
    # ── Row 1: both sorting-key distributions overlaid ────────────────────────
    plot_sorting_key(w_trop_elnino, axes[1, 0], base_level=500,
                     time_slice=t_elnino, color="steelblue")
    plot_sorting_key(w_trop_lanina, axes[1, 0], base_level=500,
                     time_slice=t_lanina, color="steelblue")

    # Retroactively style the two lines: first call → solid, second call → dashed
    lines = axes[1, 0].get_lines()
    lines[-2].set_linestyle("-");  lines[-2].set_label("El Niño")
    lines[-1].set_linestyle("--"); lines[-1].set_alpha(0.6); lines[-1].set_label("La Niña")
    axes[1, 0].legend(fontsize=8)

    plot_sorting_key(q_trop_elnino, axes[1, 1], base_level=850,
                     time_slice=t_elnino, color="darkorange")
    plot_sorting_key(q_trop_lanina, axes[1, 1], base_level=850,
                     time_slice=t_lanina, color="darkorange")

    lines = axes[1, 1].get_lines()
    lines[-2].set_linestyle("-");  lines[-2].set_label("El Niño")
    lines[-1].set_linestyle("--"); lines[-1].set_alpha(0.6); lines[-1].set_label("La Niña")
    axes[1, 1].legend(fontsize=8)

    for col in range(2):
        axes[0, col].set_xlim(0, 100)
        axes[1, col].set_xlim(0, 100)

    plt.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    for col in range(2):
        pos_top = axes[0, col].get_position()
        pos_bot = axes[1, col].get_position()
        axes[1, col].set_position([
            pos_top.x0,
            pos_bot.y0,
            pos_top.width,
            pos_bot.height,
        ])

    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# In[23]:


plot_4panel_difference(
    q_sorted_by_w500_2016_tropics, q_sorted_by_w500_2008_tropics,
    w_sorted_by_qLT_2016_tropics,  w_sorted_by_qLT_2008_tropics,
    w_trop, w_trop,
    q_trop, q_trop,
    t2016, t2008,
)


# In[24]:


print(q_sorted_by_w500_2016_tropics.dims, q_sorted_by_w500_2016_tropics.shape)
print(q_sorted_by_w500_2008_tropics.dims, q_sorted_by_w500_2008_tropics.shape)

print(q_sorted_by_w500_2016_tropics.coords)
print(q_sorted_by_w500_2008_tropics.coords)


# # El Nino composite

# In[25]:


# event year list from: https://psl.noaa.gov/enso/past_events.html


# In[26]:


ElNinoYrs = [1983, 1987, 1988, 1992, 1995, 1998, 2003, 2007, 2010, 2016]

# Build a boolean mask selecting all months in El Niño years
elnino_mask = w.time.dt.year.isin(ElNinoYrs).compute()
elnino_times = w.time.isel(time=elnino_mask)
print(f"El Niño months: {int(elnino_mask.sum())} "
      f"({ElNinoYrs[0]}–{ElNinoYrs[-1]})")

# Sort q profiles by w500 — El Niño composite, tropics
q_sorted_elnino_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = elnino_mask,        # ← pass the boolean mask directly
    lat_bounds = (-30, 30),
    output_file= "q_sorted_by_w500_elnino_tropics.nc",
)

# Sort w profiles by lower-tropospheric q — El Niño composite, tropics
w_sorted_elnino_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800., 825., 850., 875., 900., 925., 950., 975., 1000.],
    time_slice = elnino_mask,
    lat_bounds = (-30, 30),
    output_file= "w_sorted_by_qLT_elnino_tropics.nc",
)


# In[27]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],   # top row twice the height of bottom row
    hspace=0.15,            # vertical space between rows
    wspace=0.25,            # horizontal space between columns
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── Row 0: sorted profiles ────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    q_sorted_elnino_tropics, axes[0, 0],
    title  = "q sorted by w$_{500}$ — El Nino",
    xlabel = "",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0, 0], label="q (kg kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    w_sorted_elnino_tropics, axes[0, 1],
    title  = "w sorted by qLT — El Nino",
    xlabel = "",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="w (Pa s⁻¹)", pad=0.02)

# remove x tick labels on top row so bottom row x-axis is the reference
for ax in axes[0]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# ── Row 1: sorting keys ───────────────────────────────────────────────────────
plot_sorting_key(w_trop, axes[1, 0], base_level=500, time_slice=t2016, color="steelblue")
plot_sorting_key(q_trop, axes[1, 1], base_level=850, time_slice=t2016, color="darkorange")

# enforce identical x limits across all columns so percentiles align
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    axes[1, col].set_xlim(0, 100)

plt.suptitle("Sorted vertical profiles and sorting key distributions — ERA5 (El Nino Composite) : Tropics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

# ── force same left/right edges for both columns so widths truly match ────────
# (colorbar on top row shifts its axes; this corrects for that)
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0,           # align left edge
        pos_bot.y0,           # keep vertical position
        pos_top.width,        # match width exactly
        pos_bot.height,       # keep its own height
    ])

plt.savefig("4panel_sorted_profiles_composite_tropics_ElNino.png", dpi=150, bbox_inches="tight")
plt.show()


# In[28]:


# ElNinoYrs = [1983, 1987, 1988, 1992, 1995, 1998, 2003, 2007, 2010, 2016]
LaNinaYrs = [1974, 1976, 1989, 1999, 2000, 2008, 2011, 2012, 2021, 2022]

# Build a boolean mask selecting all months in El Niño years
lanina_mask = w.time.dt.year.isin(LaNinaYrs).compute()
lanina_times = w.time.isel(time=lanina_mask)
print(f"La Niña months: {int(lanina_mask.sum())} "
      f"({LaNinaYrs[0]}–{LaNinaYrs[-1]})")

# Sort q profiles by w500 — La Niña composite, tropics
q_sorted_lanina_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = lanina_mask,        # ← pass the boolean mask directly
    lat_bounds = (-30, 30),
    output_file= "q_sorted_by_w500_lanina_tropics.nc",
)

# Sort w profiles by lower-tropospheric q — El Niño composite, tropics
w_sorted_lanina_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800., 825., 850., 875., 900., 925., 950., 975., 1000.],
    time_slice = lanina_mask,
    lat_bounds = (-30, 30),
    output_file= "w_sorted_by_qLT_lanina_tropics.nc",
)


# In[29]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],   # top row twice the height of bottom row
    hspace=0.15,            # vertical space between rows
    wspace=0.25,            # horizontal space between columns
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── Row 0: sorted profiles ────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    q_sorted_lanina_tropics, axes[0, 0],
    title  = "q sorted by w$_{500}$ — La Nina",
    xlabel = "",
    cmap   = "RdBu_r",
)
fig.colorbar(pcm1, ax=axes[0, 0], label="q (kg kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    w_sorted_lanina_tropics, axes[0, 1],
    title  = "w sorted by qLT — La Nina",
    xlabel = "",
    cmap   = "RdBu_r",
    center = 0,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="w (Pa s⁻¹)", pad=0.02)

# remove x tick labels on top row so bottom row x-axis is the reference
for ax in axes[0]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# ── Row 1: sorting keys ───────────────────────────────────────────────────────
plot_sorting_key(w_trop, axes[1, 0], base_level=500, time_slice=t2008, color="steelblue")
plot_sorting_key(q_trop, axes[1, 1], base_level=850, time_slice=t2008, color="darkorange")

# enforce identical x limits across all columns so percentiles align
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    axes[1, col].set_xlim(0, 100)

plt.suptitle("Sorted vertical profiles and sorting key distributions — ERA5 (La Nina Composite) : Tropics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

# ── force same left/right edges for both columns so widths truly match ────────
# (colorbar on top row shifts its axes; this corrects for that)
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0,           # align left edge
        pos_bot.y0,           # keep vertical position
        pos_top.width,        # match width exactly
        pos_bot.height,       # keep its own height
    ])

plt.savefig("4panel_sorted_profiles_composite_tropics_LaNina.png", dpi=150, bbox_inches="tight")
plt.show()


# In[30]:


plot_4panel_difference(
    q_sorted_elnino_tropics, q_sorted_lanina_tropics,
    w_sorted_elnino_tropics,  w_sorted_lanina_tropics,
    w_trop, w_trop,
    q_trop, q_trop,
    t2016, t2008,
)


# In[ ]:





# # OLD10 vs NEW10
# ### 1941-1950 vs 2011-2020

# In[31]:


#OLD


# In[32]:


OldYrs = [1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950]

# Build a boolean mask selecting all months in Old years
old_mask = w.time.dt.year.isin(OldYrs).compute()
old_times = w.time.isel(time=old_mask)
print(f"OLD months: {int(old_mask.sum())} "
      f"({OldYrs[0]}–{OldYrs[-1]})")

# Sort q profiles by w500 — Old composite, tropics
q_sorted_old_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = old_mask,        # ← pass the boolean mask directly
    lat_bounds = (-30, 30),
    output_file= "q_sorted_by_w500_old_tropics.nc",
)

# Sort w profiles by lower-tropospheric q — Old composite, tropics
w_sorted_old_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800., 825., 850., 875., 900., 925., 950., 975., 1000.],
    time_slice = old_mask,
    lat_bounds = (-30, 30),
    output_file= "w_sorted_by_qLT_old_tropics.nc",
)


# In[33]:


# New


# In[34]:


NewYrs = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

# Build a boolean mask selecting all months in New years
new_mask = w.time.dt.year.isin(NewYrs).compute()
new_times = w.time.isel(time=new_mask)
print(f"NEW months: {int(new_mask.sum())} "
      f"({NewYrs[0]}–{NewYrs[-1]})")

# Sort q profiles by w500 — New composite, tropics
q_sorted_new_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = q,
    base_level = 500,
    time_slice = new_mask,        # ← pass the boolean mask directly
    lat_bounds = (-30, 30),
    output_file= "q_sorted_by_w500_new_tropics.nc",
)

# Sort w profiles by lower-tropospheric q — New composite, tropics
w_sorted_new_tropics = sort_profiles_by_level(
    da_base    = q,
    da_target  = w,
    base_level = [800., 825., 850., 875., 900., 925., 950., 975., 1000.],
    time_slice = new_mask,
    lat_bounds = (-30, 30),
    output_file= "w_sorted_by_qLT_new_tropics.nc",
)


# In[35]:


plot_4panel_difference(
    q_sorted_new_tropics, q_sorted_old_tropics,
    w_sorted_new_tropics,  w_sorted_old_tropics,
    w_trop, w_trop,
    q_trop, q_trop,
    t2016, t2008,
    suptitle="Sorted vertical profiles — New10 (ElNino) minus Old10 (LaNina) : Tropics",
)


# In[ ]:





# In[36]:


# Sort w profiles by w500 — New composite, tropics

# WARMER = NEW = El Nino
# COLDER = OLD = La Nina

#New
ww_sorted_new_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = w,
    base_level = 500,
    time_slice = new_mask,
    lat_bounds = (-30, 30),
    output_file= "ww_sorted_by_w500_new_tropics.nc",
)

#El Nino
ww_sorted_elnino_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = w,
    base_level = 500,
    time_slice = elnino_mask,
    lat_bounds = (-30, 30),
    output_file= "ww_sorted_by_w500_elnino_tropics.nc",
)

#Old
ww_sorted_old_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = w,
    base_level = 500,
    time_slice = old_mask,
    lat_bounds = (-30, 30),
    output_file= "ww_sorted_by_w500_old_tropics.nc",
)

#La Nina
ww_sorted_lanina_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = w,
    base_level = 500,
    time_slice = lanina_mask,
    lat_bounds = (-30, 30),
    output_file= "ww_sorted_by_w500_lanina_tropics.nc",
)


# In[37]:


fig = plt.figure(figsize=(14, 10))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[2, 1],
    hspace=0.35,
    wspace=0.25,
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    # [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── compute differences (time-mean first, then diff) ─────────────────────────
diff_enso   = ww_sorted_elnino_tropics.mean("time") - ww_sorted_lanina_tropics.mean("time")
diff_warmcold = ww_sorted_new_tropics.mean("time")  - ww_sorted_old_tropics.mean("time")

# ── symmetric colorbar limit across both panels ───────────────────────────────
vmax = np.nanpercentile(
    np.abs(np.stack([diff_enso.values, diff_warmcold.values])), 98
)
norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

# ── helper ────────────────────────────────────────────────────────────────────
def plot_diff_pcolormesh(da_diff, ax, title):
    levels = da_diff.level.values
    pcts   = da_diff.percentile.values
    pcm = ax.pcolormesh(pcts, levels, da_diff.values,
                        cmap="RdBu_r", norm=norm, shading="auto")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.tick_params(labelbottom=True)
    # ascent/descent annotations
    ax_kw = dict(transform=ax.transAxes, fontsize=9, va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
    ax.text(0.02, 0.5, "Ascent\n(−w)", ha="left",  **ax_kw)
    ax.text(0.98, 0.5, "Descent\n(+w)", ha="right", **ax_kw)
    return pcm

def plot_diff_contourf(da_diff, ax, title):
    pcts   = da_diff.percentile.values
    press  = da_diff.level.values
    clevels = np.linspace(-0.005, 0.005, 21)

    pcm = ax.contourf(pcts, press, da_diff.values,
                      levels=clevels,
                      cmap="RdBu_r",
                      extend="both")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.tick_params(labelbottom=True)
    ax_kw = dict(transform=ax.transAxes, fontsize=9, va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
    ax.text(0.02, 0.5, "Ascent\n(−w)", ha="left",  **ax_kw)
    ax.text(0.98, 0.5, "Descent\n(+w)", ha="right", **ax_kw)
    return pcm

# ── row 0: difference panels ──────────────────────────────────────────────────
pcm1 = plot_diff_contourf(diff_enso,     axes[0, 0], "El Niño − La Niña  |  w sorted by w$_{500}$")
pcm2 = plot_diff_contourf(diff_warmcold, axes[0, 1], "Warmer − Colder  |  w sorted by w$_{500}$")

# shared colorbar
cbar = fig.colorbar(pcm1, ax=axes[0, :].tolist(), 
                    label="Δw (Pa s⁻¹)", pad=0.02, shrink=0.8)

# # ── row 1: reference profiles for context ────────────────────────────────────
# def plot_ref_lines(ax, da_list, labels, colors, title):
#     """Plot time-mean w at a fixed level (500 hPa) across percentiles."""
#     for da, label, color in zip(da_list, labels, colors):
#         mean_w500 = da.sel(level=500, method="nearest").mean("time")
#         ax.plot(da.percentile.values, mean_w500.values,
#                 label=label, color=color, lw=1.8)
#     ax.axhline(0, color="k", lw=0.8, ls="--")
#     ax.axvline(50, color="grey", lw=0.8, ls=":")
#     ax.set_xlabel("Percentile of w$_{500}$", fontsize=11)
#     ax.set_ylabel("w$_{500}$ (Pa s⁻¹)", fontsize=11)
#     ax.set_title(title, fontsize=11, fontweight="bold")
#     ax.legend(fontsize=9)
#     ax.set_xlim(0, 99)

# plot_ref_lines(
#     axes[1, 0],
#     da_list = [ww_sorted_elnino_tropics, ww_sorted_lanina_tropics],
#     labels  = ["El Niño", "La Niña"],
#     colors  = ["tomato", "steelblue"],
#     title   = "w$_{500}$ by percentile — ENSO composites",
# )

# plot_ref_lines(
#     axes[1, 1],
#     da_list = [ww_sorted_new_tropics, ww_sorted_old_tropics],
#     labels  = ["Warmer", "Colder"],
#     colors  = ["tomato", "steelblue"],
#     title   = "w$_{500}$ by percentile — Warmer/Colder composites",
# )

# ── align panel widths (colorbar shifts top panels) ──────────────────────────
for col in range(2):
    axes[0, col].set_xlim(0, 100)
    # axes[1, col].set_xlim(0, 100)

plt.tight_layout()

for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[0, col].get_position()
    axes[0, col].set_position([
        pos_top.x0, pos_bot.y0, pos_top.width, pos_bot.height
    ])

plt.suptitle("Sorted w profiles — tropical composites (−30° to 30°)",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("ww_sorted_differences_tropics.png", dpi=150, bbox_inches="tight")
plt.show()


# In[ ]:





# # MSE contrast analysis
# 
# ### I am seeing a restructuring of vertical structure. Does it mean more detrainment in the middle troposphere? Does it mean more congestus? Is it because of larger or smaller contrast between cloudy and dry regions in the atmosphere? To check it, I am plotting MSE contrast between 0-10% and 10-90% for warmer and colder climate.

# In[38]:


# ── compute MSE ───────────────────────────────────────────────────────────────
# h = Cp*T + z + Lv*q
# Cp = 1005 J kg-1 K-1 (dry air)
# Lv = 2.5e6 J kg-1 (latent heat of vaporisation)
# z is already geopotential (m2 s-2), so gz term is just z

# t=ds.t #Temperature units :K
# z=ds.z #Geopotential units :m**2 s**-2

Cp = 1005.0    # J kg-1 K-1
Lv = 2.501e6   # J kg-1

mse = Cp * ds.t + ds.z + Lv * ds.q   # units: J kg-1
mse.name = "mse"
mse.attrs = {
    "long_name": "Moist Static Energy",
    "units":     "J kg-1",
    "formula":   "Cp*T + z + Lv*q",
}


# In[39]:


# ── sort MSE profiles by w500 — four composites, tropics ─────────────────────
mse_sorted_new_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = mse,
    base_level = 500,
    time_slice = new_mask,
    lat_bounds = (-30, 30),
    output_file= "mse_sorted_by_w500_new_tropics.nc",
)

mse_sorted_elnino_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = mse,
    base_level = 500,
    time_slice = elnino_mask,
    lat_bounds = (-30, 30),
    output_file= "mse_sorted_by_w500_elnino_tropics.nc",
)

mse_sorted_old_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = mse,
    base_level = 500,
    time_slice = old_mask,
    lat_bounds = (-30, 30),
    output_file= "mse_sorted_by_w500_old_tropics.nc",
)

mse_sorted_lanina_tropics = sort_profiles_by_level(
    da_base    = w,
    da_target  = mse,
    base_level = 500,
    time_slice = lanina_mask,
    lat_bounds = (-30, 30),
    output_file= "mse_sorted_by_w500_lanina_tropics.nc",
)


# In[40]:


def mse_contrast(da_sorted):
    """
    ΔMSE = <h>_p90-100 − <h>_p10-90
    Ascent region   : percentile  0–10  (most negative w = strongest ascent)
    Descent region  : percentile 90–100 (most positive w = strongest descent)
    Middle          : percentile 10–90
    Returns a DataArray of shape (level,)
    """
    h = da_sorted.mean("time")   # (level, percentile)
    ascent  = h.sel(percentile=slice(0,  10)).mean("percentile")
    descent = h.sel(percentile=slice(90, 100)).mean("percentile")
    return descent - ascent      # shape (level,)


# ── compute contrasts ─────────────────────────────────────────────────────────
contrast_elnino  = mse_contrast(mse_sorted_elnino_tropics)
contrast_lanina  = mse_contrast(mse_sorted_lanina_tropics)
contrast_new     = mse_contrast(mse_sorted_new_tropics)
contrast_old     = mse_contrast(mse_sorted_old_tropics)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

def plot_contrast_pair(ax, da1, da2, label1, label2, color1, color2, title):
    levels = da1.level.values
    ax.plot(da1.values / 1e3, levels, color=color1, lw=2,   label=label1)
    ax.plot(da2.values / 1e3, levels, color=color2, lw=2,   label=label2)
    ax.plot((da1 - da2).values / 1e3, levels,
            color="k", lw=1.5, ls="--", label=f"{label1} − {label2}")
    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("ΔMSE  (kJ kg⁻¹)", fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

plot_contrast_pair(
    axes[0],
    contrast_elnino, contrast_lanina,
    "El Niño", "La Niña",
    "tomato", "steelblue",
    "ΔMSE contrast — ENSO\n(descent p90–100 minus ascent p0–10)",
)

plot_contrast_pair(
    axes[1],
    contrast_new, contrast_old,
    "Warmer", "Colder",
    "tomato", "steelblue",
    "ΔMSE contrast — Warmer/Colder\n(descent p90–100 minus ascent p0–10)",
)

plt.suptitle("MSE contrast between ascent and descent regions — tropics (−30° to 30°)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("mse_contrast_tropics.png", dpi=150, bbox_inches="tight")
plt.show()


# In[41]:


fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1, 1],
    hspace=0.35,
    wspace=0.15,
)
axes = np.array([
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
])

# ── compute a shared vmin/vmax across all four panels ────────────────────────
all_data = np.concatenate([
    mse_sorted_lanina_tropics.mean("time").values.ravel(),
    mse_sorted_elnino_tropics.mean("time").values.ravel(),
    mse_sorted_old_tropics.mean("time").values.ravel(),
    mse_sorted_new_tropics.mean("time").values.ravel(),
])
vmin = np.nanpercentile(all_data, 2)
vmax = np.nanpercentile(all_data, 98)

# ── row 0: ENSO ───────────────────────────────────────────────────────────────
pcm1 = plot_sorted_profiles(
    mse_sorted_lanina_tropics, axes[0, 0],
    title="MSE sorted by w$_{500}$ — La Niña",
    xlabel="", cmap="YlOrRd", vmin=vmin, vmax=vmax,
)
fig.colorbar(pcm1, ax=axes[0, 0], label="MSE (J kg⁻¹)", pad=0.02)

pcm2 = plot_sorted_profiles(
    mse_sorted_elnino_tropics, axes[0, 1],
    title="MSE sorted by w$_{500}$ — El Niño",
    xlabel="", cmap="YlOrRd", vmin=vmin, vmax=vmax,
)
fig.colorbar(pcm2, ax=axes[0, 1], label="MSE (J kg⁻¹)", pad=0.02)

# ── row 1: warm/cold ──────────────────────────────────────────────────────────
pcm3 = plot_sorted_profiles(
    mse_sorted_old_tropics, axes[1, 0],
    title="MSE sorted by w$_{500}$ — Colder",
    xlabel="", cmap="YlOrRd", vmin=vmin, vmax=vmax,
)
fig.colorbar(pcm3, ax=axes[1, 0], label="MSE (J kg⁻¹)", pad=0.02)

pcm4 = plot_sorted_profiles(
    mse_sorted_new_tropics, axes[1, 1],
    title="MSE sorted by w$_{500}$ — Warmer",
    xlabel="", cmap="YlOrRd", vmin=vmin, vmax=vmax,
)
fig.colorbar(pcm4, ax=axes[1, 1], label="MSE (J kg⁻¹)", pad=0.02)

# ── shared x label on bottom row only ────────────────────────────────────────
for ax in axes[0]:
    ax.tick_params(labelbottom=False)
for ax in axes[1]:
    ax.set_xlabel("Percentile of w$_{500}$", fontsize=11)

# ── enforce identical x limits ────────────────────────────────────────────────
for ax in axes.ravel():
    ax.set_xlim(0, 100)

plt.suptitle("MSE sorted by w$_{500}$ — Tropical composites (−30° to 30°)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

# ── align panel widths after tight_layout (colorbars shift axes) ──────────────
for col in range(2):
    pos_top = axes[0, col].get_position()
    pos_bot = axes[1, col].get_position()
    axes[1, col].set_position([
        pos_top.x0, pos_bot.y0, pos_top.width, pos_bot.height
    ])

plt.savefig("4panel_mse_sorted_composites_tropics.png", dpi=150, bbox_inches="tight")
plt.show()


# In[42]:


fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2, wspace=0.2)
axes = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])])

# ── compute differences ───────────────────────────────────────────────────────
diff_enso     = mse_sorted_elnino_tropics.mean("time") - mse_sorted_lanina_tropics.mean("time")
diff_warmcold = mse_sorted_new_tropics.mean("time")    - mse_sorted_old_tropics.mean("time")

# ── shared symmetric contour levels ──────────────────────────────────────────
vmax = np.nanpercentile(
    np.abs(np.stack([diff_enso.values, diff_warmcold.values])), 90
)
clevels = np.linspace(-vmax, vmax, 21)
clevels1 = np.linspace(-vmax/2, vmax/2, 21)

# ── plot ──────────────────────────────────────────────────────────────────────
def plot_mse_diff(da_diff, ax, title, cl):
    pcts  = da_diff.percentile.values
    press = da_diff.level.values
    pcm = ax.contourf(pcts, press, da_diff.values,
                      levels=cl, cmap="RdBu_r", extend="both")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile of w$_{500}$", fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax_kw = dict(transform=ax.transAxes, fontsize=9, va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
    ax.text(0.02, 0.5, "Ascent\n(−w)", ha="left",  **ax_kw)
    ax.text(0.98, 0.5, "Descent\n(+w)", ha="right", **ax_kw)
    return pcm

pcm1 = plot_mse_diff(diff_enso,     axes[0], "ΔMSE — El Niño − La Niña",clevels1)
pcm2 = plot_mse_diff(diff_warmcold, axes[1], "ΔMSE — Warmer − Colder",clevels)

# ── shared colorbar ───────────────────────────────────────────────────────────
fig.colorbar(pcm1, ax=axes.tolist(), label="ΔMSE (J kg⁻¹)", pad=0.02, shrink=0.8)

plt.suptitle("MSE difference sorted by w$_{500}$ — tropics (−30° to 30°) (Notice: scale of ENSO)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("mse_diff_sorted_tropics.png", dpi=150, bbox_inches="tight")
plt.show()


# # GMS

# In[43]:


# ── Partial GMS using omega (Back & Bretherton 2006 approach) ─────────────────
# 
# Standard definition:
#   PGMS(p*) = ∫[p*→ptop] w * (∂h/∂p) dp  /  ∫[p*→ptop] w * (∂f/∂p) dp
#
# But the most tractable form with sorted profiles:
#   PGMS(p*) = -∫[ptop→p*] <w·h> dp  /  -∫[ptop→p*] <w> dp
#
# where the integral runs from TOA downward to level p*
# Negative signs because omega is positive downward
#
# We need joint wh — so we sort wh profiles by w500
# (cannot derive this from separately sorted w and h)

# ── Step 1: compute w*h at each level and cell ────────────────────────────────
wh = ds.w * mse      # (time, level, cell) units: Pa s-1 * J kg-1
wh.name = "wh"
wh.attrs = {
    "long_name": "Vertical MSE flux (omega * MSE)",
    "units":     "Pa s-1 J kg-1",
}


# In[ ]:


# ── Step 2: sort wh and w profiles by w500 — same four composites ─────────────
# We need BOTH wh and w sorted by the SAME w500 key so they share percentile bins

wh_sorted_elnino_tropics = sort_profiles_by_level(
    da_base    = ds.w,
    da_target  = wh,
    base_level = 500,
    time_slice = elnino_mask,
    lat_bounds = (-30, 30),
    output_file= "wh_sorted_by_w500_elnino_tropics.nc",
)

wh_sorted_lanina_tropics = sort_profiles_by_level(
    da_base    = ds.w,
    da_target  = wh,
    base_level = 500,
    time_slice = lanina_mask,
    lat_bounds = (-30, 30),
    output_file= "wh_sorted_by_w500_lanina_tropics.nc",
)

wh_sorted_new_tropics = sort_profiles_by_level(
    da_base    = ds.w,
    da_target  = wh,
    base_level = 500,
    time_slice = new_mask,
    lat_bounds = (-30, 30),
    output_file= "wh_sorted_by_w500_new_tropics.nc",
)

wh_sorted_old_tropics = sort_profiles_by_level(
    da_base    = ds.w,
    da_target  = wh,
    base_level = 500,
    time_slice = old_mask,
    lat_bounds = (-30, 30),
    output_file= "wh_sorted_by_w500_old_tropics.nc",
)

# I already have ww_sorted_* (w sorted by w500) from before — reuse those


# In[ ]:


# ── Step 3: compute PGMS profile from sorted arrays ───────────────────────────
def compute_pgms(wh_sorted, ww_sorted, ascent_pct=slice(0, 10)):
    """
    Partial GMS profile for ascent region.

    PGMS(p*) = cumulative integral of <wh> from TOA to p*
             / cumulative integral of <w>  from TOA to p*

    Both integrals run top-down (TOA → surface) so we flip level order,
    integrate, then flip back.

    Parameters
    ----------
    wh_sorted : DataArray (time, level, percentile) — w*MSE sorted by w500
    ww_sorted : DataArray (time, level, percentile) — w sorted by w500
    ascent_pct: percentile slice selecting ascent columns

    Returns
    -------
    DataArray (level,) — PGMS profile in ascent region
    """
    # time-mean and select ascent percentiles
    wh = wh_sorted.mean("time").sel(percentile=ascent_pct).mean("percentile")  # (level,)
    ww = ww_sorted.mean("time").sel(percentile=ascent_pct).mean("percentile")  # (level,)

    # pressure in Pa for integration (level coord is in hPa)
    p_Pa = wh.level.values * 100.0   # hPa → Pa

    # flip to integrate top-down (TOA first)
    wh_vals = wh.values[::-1]
    ww_vals = ww.values[::-1]
    p_vals  = p_Pa[::-1]

    # cumulative trapezoid integral top → p*
    from scipy.integrate import cumulative_trapezoid
    cum_wh = cumulative_trapezoid(wh_vals, p_vals, initial=0)
    cum_ww = cumulative_trapezoid(ww_vals, p_vals, initial=0)

    # PGMS: avoid division by zero near TOA
    with np.errstate(invalid="ignore", divide="ignore"):
        pgms_vals = np.where(np.abs(cum_ww) > 1e-10,
                             cum_wh / cum_ww,
                             np.nan)

    # flip back to surface-up order
    pgms_vals = pgms_vals[::-1]

    pgms = xr.DataArray(
        pgms_vals,
        dims=["level"],
        coords={"level": wh.level.values},
        name="PGMS",
        attrs={
            "long_name": "Partial Gross Moist Stability (ascent region)",
            "units":     "J kg-1",
        },
    )
    return pgms


# In[ ]:


# ── Step 4: compute for all four composites ────────────────────────────────────
pgms_elnino = compute_pgms(wh_sorted_elnino_tropics, ww_sorted_elnino_tropics)
pgms_lanina = compute_pgms(wh_sorted_lanina_tropics, ww_sorted_lanina_tropics)
pgms_new    = compute_pgms(wh_sorted_new_tropics,    ww_sorted_new_tropics)
pgms_old    = compute_pgms(wh_sorted_old_tropics,    ww_sorted_old_tropics)


# In[ ]:


# ── Step 5: plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

def plot_pgms(ax, da1, da2, label1, label2, color1, color2, title):
    levels = da1.level.values
    ax.plot(da1.values / 1e3, levels, color=color1, lw=2,   label=label1)
    ax.plot(da2.values / 1e3, levels, color=color2, lw=2,   label=label2)
    ax.plot((da1 - da2).values / 1e3, levels,
            color="k", lw=1.5, ls="--", label=f"{label1} − {label2}")
    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("PGMS  (kJ kg⁻¹)", fontsize=11)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    # shade positive (stable) vs negative (unstable) regions
    ax.axvspan(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 10,
               alpha=0.05, color="red",  label="_stable")
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < 0 else -10, 0,
               alpha=0.05, color="blue", label="_unstable")

plot_pgms(
    axes[0],
    pgms_elnino, pgms_lanina,
    "El Niño", "La Niña", "tomato", "steelblue",
    "PGMS — ENSO (ascent p0–10)",
)
plot_pgms(
    axes[1],
    pgms_new, pgms_old,
    "Warmer", "Colder", "tomato", "steelblue",
    "PGMS — Warmer/Colder (ascent p0–10)",
)

plt.suptitle("Partial Gross Moist Stability — tropics (−30° to 30°)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("pgms_profiles_tropics.png", dpi=150, bbox_inches="tight")
plt.show()

