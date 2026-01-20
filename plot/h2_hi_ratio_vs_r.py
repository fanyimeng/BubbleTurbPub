#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H2/HI ratio vs radius with shaded radial band and fixed red bubbles (no data-driven coloring).

Features
--------
1) Pixel-based radial profile of H2/HI volume density ratio:
   - Radius map from '../data/r-m31-jcomb_modHeader.fits' (pc).
   - H2/HI ratio map from '../data/M31_H2_to_HI_ratio.fits' (dimensionless).
   - In a given R range [rmin_kpc, rmax_kpc), binned in radius (default 1 kpc).
   - For each bin, compute median and lower/upper percentiles (default 25/75).
   - Plot as a shadedM31_H2_to_HI_ratio.fits band (between p_lo and p_hi) plus a median line.
   - y-axis in log scale.

2) Superbubble points:
   - Bubble catalog '../code/1113.tab' (fixed width), with at least columns:
       'ra_pix', 'dec_pix', 'major', 'minor', 'pa', 'radius_pc',
       (n_HI columns are ignored for plotting; bubbles are fixed red).
   - For each bubble:
       - Use the same ring definition as in compute_bubble_nhi.py:
         inner ellipse (major, minor), outer ellipse (ring_scale*major, ring_scale*minor).
         Ring region = outer minus inner.
       - Sample H2/HI ratio map in the ring region.
       - Bubble y-value = median(H2/HI) in the ring.
       - Bubble y-error: from p_lo/p_hi percentiles in the ring (asymmetric error).
       - Bubble x-position = center R from radius map (pc -> kpc).
       - Bubble x-error: same geometry as EnergyBudget script:
           h_pc = 182 + 16 * R_kpc
           dh   = h_pc * tan(77 deg) * cos(alpha - 90 deg)
           dr   = radius_pc
           dx_kpc = sqrt(dh^2 + dr^2) / 1e3
   - Bubble points are fixed red, edge-free, with no colorbar.

Inputs (defaults)
-----------------
RADIUS_FITS_PATH = "../data/r-m31-jcomb_modHeader.fits"       (pc)
RATIO_FITS_PATH  = "../data/M31_H2_to_HI_ratio.fits"          (dimensionless, H2/HI)
ALPHA_FITS_PATH  = "../data/alpha-m31-jcomb_modHeader.fits"   (deg)
BUBBLE_TAB_PATH  = "../code/1113.tab"

Output
------
FIGURE_PDF = "H2_HI_ratio_vs_R_band_with_bubbles.pdf"
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# -----------------------
# Default paths and parameters
# -----------------------

RADIUS_FITS_PATH = "../data/r-m31-jcomb_modHeader.fits"         # pc
RATIO_FITS_PATH  = "../data/M31_H2_to_HI_ratio.fits"            # H2/HI
ALPHA_FITS_PATH  = "../data/alpha-m31-jcomb_modHeader.fits"     # deg
BUBBLE_TAB_PATH  = "../code/1113.tab"

FIGURE_PDF       = "h2_hi_ratio_vs_r.pdf"

# Radius range and bin width
RMIN_KPC      = 3.0
RMAX_KPC      = 18.0
BIN_WIDTH_KPC = 0.5

# Percentiles for vertical statistics
P_LO = 25.0
P_HI = 75.0

# Minimum number of pixels per radial bin / ring to keep
MIN_PIXELS_PER_BIN  = 30
MIN_PIXELS_PER_RING = 30

# Ring geometry for bubbles (outer ellipse factor)
RING_SCALE = 2.0

# Bubble coloring
NRING_COLUMN         = "n_HI_ring_cm-3"
COLOR_CMAP_NAME      = "coolwarm"
COLORBAR_LABEL       = r"$n_{\rm HI}$ [cm$^{-3}$]"
COLORBAR_ASPECT      = 40

BUBBLE_MARKER_SIZE = 45
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)
BUBBLE_FACE_ALPHA = 0.7


# -----------------------
# Helper functions
# -----------------------

def print_minmax(name, arr):
    arr = np.asarray(arr, dtype=float)
    m = np.isfinite(arr)
    if not np.any(m):
        print(f"[H2/HI vs R] {name}: no finite values")
        return
    print(
        f"[H2/HI vs R] {name}: min={np.nanmin(arr[m]):.6g}, "
        f"max={np.nanmax(arr[m]):.6g}, N={np.count_nonzero(m)}"
    )


def radial_bin_stats(
    r_kpc,
    values,
    rmin_kpc,
    rmax_kpc,
    bin_width_kpc,
    p_lo=25.0,
    p_hi=75.0,
    min_pixels_per_bin=1
):
    """
    Bin (r_kpc, values) in radius and compute median and asymmetric errors
    based on percentiles p_lo / p_hi.

    Returns:
        centers   : bin centers [kpc]
        med_vals  : median values
        err_lower : med - p_lo
        err_upper : p_hi - med
        counts    : number of pixels per bin
    """
    r_kpc = np.asarray(r_kpc, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(r_kpc) & np.isfinite(values)
    r_kpc = r_kpc[mask]
    values = values[mask]

    print_minmax("R_kpc (all finite)", r_kpc)
    print_minmax("ratio H2/HI (all finite)", values)

    nbins = int(np.floor((rmax_kpc - rmin_kpc) / bin_width_kpc))
    if nbins <= 0:
        raise ValueError("Invalid binning: ensure rmax_kpc > rmin_kpc and bin_width_kpc > 0")

    edges = rmin_kpc + np.arange(nbins + 1) * bin_width_kpc
    centers = 0.5 * (edges[:-1] + edges[1:])

    med_values = np.full(nbins, np.nan, dtype=float)
    err_lower = np.full(nbins, np.nan, dtype=float)
    err_upper = np.full(nbins, np.nan, dtype=float)
    counts    = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        lo = edges[i]
        hi = edges[i + 1]
        m_bin = (r_kpc >= lo) & (r_kpc < hi)
        if not np.any(m_bin):
            continue

        vals_bin = values[m_bin]
        vals_bin = vals_bin[np.isfinite(vals_bin)]

        counts[i] = vals_bin.size
        if vals_bin.size < min_pixels_per_bin:
            continue

        med = np.nanmedian(vals_bin)
        plo = np.nanpercentile(vals_bin, p_lo)
        phi = np.nanpercentile(vals_bin, p_hi)

        med_values[i] = med
        err_lower[i]  = med - plo
        err_upper[i]  = phi - med

        print(
            f"[H2/HI vs R] Bin {i}: R=[{lo:.2f},{hi:.2f}) kpc, "
            f"N={vals_bin.size}, median={med:.4g}, "
            f"plo={plo:.4g}, phi={phi:.4g}"
        )

    valid = np.isfinite(med_values)
    return centers[valid], med_values[valid], err_lower[valid], err_upper[valid], counts[valid]


def values_at_pixels(data: np.ndarray, x_pix, y_pix) -> np.ndarray:
    """
    Sample a 2D array at (x_pix, y_pix) positions, rounding to nearest integer.
    Out-of-range samples return NaN.
    """
    data  = np.asarray(data)
    x_arr = np.asarray(x_pix, dtype=float)
    y_arr = np.asarray(y_pix, dtype=float)
    xi = np.round(x_arr).astype(int)
    yi = np.round(y_arr).astype(int)
    values = np.full(len(xi), np.nan, dtype=float)

    mask = (
        (xi >= 0) & (yi >= 0) &
        (yi < data.shape[0]) &
        (xi < data.shape[1])
    )
    values[mask] = data[yi[mask], xi[mask]]
    return values


def ring_values_for_bubble(
    data: np.ndarray,
    x0: float,
    y0: float,
    major: float,
    minor: float,
    pa_deg: float,
    ring_scale: float = 2.0
) -> np.ndarray:
    """
    Compute values of `data` inside the ring defined by:
        outer ellipse: a_out = ring_scale * major, b_out = ring_scale * minor
        inner ellipse: a_in  = major,             b_in  = minor
        ring = outer - inner.

    All lengths in pixels. pa_deg is the ellipse PA in degrees, following the same
    convention as compute_bubble_nhi.py.

    Returns a 1D array of values within the ring.
    """
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape

    if not np.isfinite([x0, y0, major, minor, pa_deg]).all():
        return np.array([], dtype=float)
    if major <= 0 or minor <= 0 or ring_scale <= 1.0:
        return np.array([], dtype=float)

    a_in  = float(major)
    b_in  = float(minor)
    a_out = float(major) * float(ring_scale)
    b_out = float(minor) * float(ring_scale)

    # Bounding box: span based on outer ellipse
    span = float(np.hypot(a_out, b_out))
    x_min = max(int(np.floor(x0 - span)), 0)
    x_max = min(int(np.ceil(x0 + span)) + 1, nx)
    y_min = max(int(np.floor(y0 - span)), 0)
    y_max = min(int(np.ceil(y0 + span)) + 1, ny)

    if x_min >= x_max or y_min >= y_max:
        return np.array([], dtype=float)

    sub = data[y_min:y_max, x_min:x_max]
    if sub.size == 0:
        return np.array([], dtype=float)

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    x_rel = xx - x0
    y_rel = yy - y0

    theta = np.deg2rad(pa_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = x_rel * cos_t + y_rel * sin_t
    y_rot = -x_rel * sin_t + y_rel * cos_t

    # Ellipse equations
    inner = (x_rot / a_in)**2 + (y_rot / b_in)**2 <= 1.0
    outer = (x_rot / a_out)**2 + (y_rot / b_out)**2 <= 1.0

    ring_mask = outer & (~inner)
    if not np.any(ring_mask):
        return np.array([], dtype=float)

    vals = sub[ring_mask]
    vals = vals[np.isfinite(vals)]
    return vals


def _auto_vmin_vmax(vals, q_lo=0.02, q_hi=0.98):
    """Robust vmin/vmax from percentiles; fallback to finite min/max."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, q_lo * 100.0)
    hi = np.nanpercentile(v, q_hi * 100.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(v)
        hi = np.nanmax(v)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    return float(lo), float(hi)


# -----------------------
# Main plotting function
# -----------------------

def plot_H2_HI_ratio_band_with_bubbles(
    radius_fits_path=RADIUS_FITS_PATH,
    ratio_fits_path=RATIO_FITS_PATH,
    alpha_fits_path=ALPHA_FITS_PATH,
    bubble_tab_path=BUBBLE_TAB_PATH,
    figure_pdf=FIGURE_PDF,
    # radial band
    rmin_kpc=RMIN_KPC,
    rmax_kpc=RMAX_KPC,
    bin_width_kpc=BIN_WIDTH_KPC,
    p_lo=P_LO,
    p_hi=P_HI,
    min_pixels_per_bin=MIN_PIXELS_PER_BIN,
    # ring geometry
    ring_scale=RING_SCALE,
    min_pixels_per_ring=MIN_PIXELS_PER_RING,
    # bubble coloring
    nring_column=NRING_COLUMN,
    color_cmap_name=COLOR_CMAP_NAME,
    colorbar_label=COLORBAR_LABEL,
    colorbar_aspect=COLORBAR_ASPECT,
    color_vmin=None,
    color_vmax=None,
):
    """
    Plot H2/HI ratio vs radius with:
      - Pixel-based radial band (p_lo/p_hi percentiles) and median line.
      - Bubble points (ring medians) with x and y errorbars; fixed red markers (no colorbar).
    """
    # --- Read maps ---

    with fits.open(radius_fits_path) as hdul_r:
        r_pc_map = hdul_r[0].data.astype(float)

    with fits.open(ratio_fits_path) as hdul_ratio:
        ratio_map = hdul_ratio[0].data.astype(float)

    with fits.open(alpha_fits_path) as hdul_a:
        alpha_map = hdul_a[0].data.astype(float)

    if r_pc_map.shape != ratio_map.shape or r_pc_map.shape != alpha_map.shape:
        raise ValueError(
            f"Shape mismatch among radius {r_pc_map.shape}, ratio {ratio_map.shape}, "
            f"alpha {alpha_map.shape}. Ensure they are on the same grid."
        )

    # --- Pixel-based radial profile ---

    r_pc_flat    = r_pc_map.ravel()
    ratio_flat   = ratio_map.ravel()
    r_kpc_flat   = r_pc_flat / 1.0e3

    mpix = (
        np.isfinite(r_kpc_flat)
        & np.isfinite(ratio_flat)
        & (ratio_flat > 0.0)
        & (r_kpc_flat >= rmin_kpc)
        & (r_kpc_flat < rmax_kpc)
    )

    r_use     = r_kpc_flat[mpix]
    ratio_use = ratio_flat[mpix]

    print_minmax("R_kpc (used, pixels)", r_use)
    print_minmax("H2/HI ratio (used, pixels)", ratio_use)

    centers, med_vals, err_lo, err_hi, counts = radial_bin_stats(
        r_use,
        ratio_use,
        rmin_kpc=rmin_kpc,
        rmax_kpc=rmax_kpc,
        bin_width_kpc=bin_width_kpc,
        p_lo=p_lo,
        p_hi=p_hi,
        min_pixels_per_bin=min_pixels_per_bin,
    )

    # For log y-axis, enforce positive medians
    pos = med_vals > 0
    centers  = centers[pos]
    med_vals = med_vals[pos]
    err_lo   = err_lo[pos]
    err_hi   = err_hi[pos]

    # Ensure lower error does not push below zero
    err_lo = np.minimum(err_lo, 0.9 * med_vals)

    y_band_low  = med_vals - err_lo
    y_band_high = med_vals + err_hi

    # --- Bubble points ---

    df_bub = pd.read_fwf(bubble_tab_path, header=0, infer_nrows=int(1e6))

    required_cols = {"ra_pix", "dec_pix", "major", "minor", "pa", "radius_pc"}
    missing = required_cols - set(df_bub.columns)
    if missing:
        raise KeyError(f"{bubble_tab_path} missing required columns: {sorted(missing)}")

    ra_pix    = df_bub["ra_pix"].to_numpy(dtype=float)
    dec_pix   = df_bub["dec_pix"].to_numpy(dtype=float)
    major_pix = df_bub["major"].to_numpy(dtype=float)
    minor_pix = df_bub["minor"].to_numpy(dtype=float)
    pa_deg    = df_bub["pa"].to_numpy(dtype=float)
    rbub_pc   = df_bub["radius_pc"].to_numpy(dtype=float)

    # Center R and alpha at bubble positions
    r_bub_pc  = values_at_pixels(r_pc_map,   ra_pix, dec_pix)
    alpha_bub = values_at_pixels(alpha_map,  ra_pix, dec_pix)
    r_bub_kpc = r_bub_pc / 1.0e3

    # H2/HI in rings for each bubble
    bub_y      = np.full_like(r_bub_kpc, np.nan, dtype=float)
    bub_y_lo   = np.full_like(r_bub_kpc, np.nan, dtype=float)
    bub_y_hi   = np.full_like(r_bub_kpc, np.nan, dtype=float)

    for i, (x0, y0, a, b, pa) in enumerate(zip(ra_pix, dec_pix, major_pix, minor_pix, pa_deg)):
        vals_ring = ring_values_for_bubble(
            ratio_map,
            x0=x0,
            y0=y0,
            major=a,
            minor=b,
            pa_deg=pa,
            ring_scale=ring_scale,
        )
        # only keep positive ratios for log
        vals_ring = vals_ring[vals_ring > 0.0]
        if vals_ring.size < min_pixels_per_ring:
            continue

        med = np.nanmedian(vals_ring)
        plo = np.nanpercentile(vals_ring, p_lo)
        phi = np.nanpercentile(vals_ring, p_hi)

        bub_y[i]    = med
        bub_y_lo[i] = med - plo
        bub_y_hi[i] = phi - med

    print_minmax("H2/HI ratio (bubbles, ring median)", bub_y)

    # x-error following EnergyBudget geometry
    h_pc    = 182.0 + 16.0 * r_bub_kpc
    angles  = alpha_bub - 90.0
    dh      = h_pc * np.tan(np.deg2rad(77.0)) * np.cos(np.deg2rad(angles))
    dr      = rbub_pc
    dx_kpc  = np.sqrt(dh**2 + dr**2) / 1.0e3

    # Valid bubbles for plotting
    mbub = (
        np.isfinite(r_bub_kpc)
        & np.isfinite(bub_y)
        & (bub_y > 0.0)
        & (r_bub_kpc >= rmin_kpc)
        & (r_bub_kpc < rmax_kpc)
    )

    r_bub_kpc_use = r_bub_kpc[mbub]
    bub_y_use     = bub_y[mbub]
    bub_y_lo_use  = bub_y_lo[mbub]
    bub_y_hi_use  = bub_y_hi[mbub]
    dx_use        = dx_kpc[mbub]
    print_minmax("R_kpc (bubbles, used)", r_bub_kpc_use)

    # Clamp y lower error
    bub_y_lo_use = np.minimum(bub_y_lo_use, 0.9 * bub_y_use)

    # --- Plotting ---

    fig, ax = plt.subplots(figsize=(4.9, 4.1))

    # Shaded radial band
    ax.fill_between(
        centers,
        y_band_low,
        y_band_high,
        color=(0.7, 0.7, 0.7, 0.35),
        edgecolor=(0.2, 0.2, 0.2, 0.0),
        linewidth=0.5,
        label="Pixel band (p25–p75)",
        zorder=1,
    )

    # Median line
    ax.plot(
        centers,
        med_vals,
        color="k",
        linewidth=1.0,
        linestyle="-",
        label="Pixel median",
        zorder=2,
    )

    # Bubble error bars in x and y
    if r_bub_kpc_use.size > 0:
        # vertical errors as asymmetric
        yerr = np.vstack([bub_y_lo_use, bub_y_hi_use])
        # draw each with matching error bars (fixed gray)
        for xi, yi, xe, ye_lo, ye_hi in zip(
            r_bub_kpc_use,
            bub_y_use,
            dx_use,
            bub_y_lo_use,
            bub_y_hi_use,
        ):
            yerr_i = np.array([[ye_lo], [ye_hi]])
            ax.errorbar(
                [xi],
                [yi],
                xerr=[[xe], [xe]],
                yerr=yerr_i,
                fmt="none",
                ecolor=(0.3, 0.3, 0.3, 0.35),
                elinewidth=0.7,
                capsize=0.0,
                zorder=3,
            )

        ax.scatter(
            r_bub_kpc_use,
            bub_y_use,
            s=BUBBLE_MARKER_SIZE,
            facecolor=(BUBBLE_BASE_FACE_COLOR[0], BUBBLE_BASE_FACE_COLOR[1], BUBBLE_BASE_FACE_COLOR[2], BUBBLE_FACE_ALPHA),
            edgecolor="none",
            linewidths=0.0,
            zorder=4,
            label="New Bubbles",
        )

    ax.set_xlabel("Distance to M31's center: R [kpc]")
    ax.set_ylabel("$N_{\\rm H_2}/N_{\\rm HI}$")
    ax.set_xlim(rmin_kpc, rmax_kpc)
    ax.set_yscale("log")



    # y-range from band and bubble points
    y_all = []
    if med_vals.size > 0:
        y_all.append(np.nanmin(y_band_low))
        y_all.append(np.nanmax(y_band_high))
    if r_bub_kpc_use.size > 0:
        y_all.append(np.nanmin(bub_y_use - bub_y_lo_use))
        y_all.append(np.nanmax(bub_y_use + bub_y_hi_use))
    y_all = np.array(y_all, dtype=float)
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    if y_all.size > 0:
        y_min = y_all.min()
        y_max = y_all.max()
        ax.set_ylim(y_min / 1.5, y_max * 1.5)

    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(alpha=0.2, linestyle=":")

    # Legend: use a neutral bubble marker in legend
    legend_items = []
    legend_labels = []

    sample_marker = Line2D(
        [], [], marker='o', linestyle='None',
        markersize=np.sqrt(45),
        markerfacecolor=(BUBBLE_BASE_FACE_COLOR[0], BUBBLE_BASE_FACE_COLOR[1], BUBBLE_BASE_FACE_COLOR[2], BUBBLE_FACE_ALPHA),
        markeredgecolor="none",
        markeredgewidth=0.0,
    )
    legend_items.append(sample_marker)
    legend_labels.append("New Bubbles")

    legend_items.append(Line2D([], [], color="k", linewidth=1.0))
    legend_labels.append("Disk average")

    ax.legend(
        legend_items, legend_labels,
        loc="lower left", fontsize=8,
        frameon=True, framealpha=0.9,
        edgecolor="black"
    )
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(5, 12)

    plt.tight_layout()
    plt.savefig(figure_pdf)
    plt.close()
    print(f"[H2/HI vs R] Figure with band and bubbles saved to: {figure_pdf}")


# -----------------------
# Script entry point
# -----------------------

if __name__ == "__main__":
    plot_H2_HI_ratio_band_with_bubbles(
        radius_fits_path=RADIUS_FITS_PATH,
        ratio_fits_path=RATIO_FITS_PATH,
        alpha_fits_path=ALPHA_FITS_PATH,
        bubble_tab_path=BUBBLE_TAB_PATH,
        figure_pdf=FIGURE_PDF,
        rmin_kpc=RMIN_KPC,
        rmax_kpc=RMAX_KPC,
        bin_width_kpc=BIN_WIDTH_KPC,
        p_lo=P_LO,
        p_hi=P_HI,
        min_pixels_per_bin=MIN_PIXELS_PER_BIN,
        ring_scale=RING_SCALE,
        min_pixels_per_ring=MIN_PIXELS_PER_RING,
        nring_column=NRING_COLUMN,
        color_cmap_name=COLOR_CMAP_NAME,
        colorbar_label=COLORBAR_LABEL,
        colorbar_aspect=COLORBAR_ASPECT,
        color_vmin=None,
        color_vmax=None,
    )
