#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scatter of v^2 / t_diss: bubble expansion vs turbulent dispersion at the same radius.

Definitions (cgs)
-----------------
- Turbulent metric: (sigma_kms^3) / (2 * h_pc * pc2cm), where
  t_diss,turb = 2 h_pc / sigma  (crossing time) and sigma corrected from mom2.
- Bubble metric: v_exp^2 / t_diss,bub with v_exp in cm/s and
  t_diss,bub = 0.6 * r_pc / v_exp (converted to seconds).
  This yields v_exp^3 / (0.6 * r_pc * pc2cm).

Inputs
------
- bubble_table_path : ../code/1113.tab (fixed-width; uses expansion_vel, radius_pc, ra_pix, dec_pix)
- ring_table_path   : ../code/1113.tab (only used to fetch n_ring if you add color; not required here)
- alpha_fits        : ../data/alpha-m31-jcomb_modHeader.fits (for alignment; radius map sampling)
- r_fits            : ../data/r-m31-jcomb_modHeader.fits (pc, for R_kpc coloring)
- prof_max_path     : ../data/profile_resampled_max.tsv (columns r, x1_mom0, x1_mom2)
- prof_min_path     : ../data/profile_resampled_min.tsv (columns r, x1_mom0, x1_mom2)

Outputs
-------
- v2_over_t_vs_turb.pdf : Bubble v^2/t vs turbulent v^2/t (both cm^2 s^-3) colored by R_kpc.
- Optional CSV export if export_table=True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

pc2cm = 3e18
COSI = np.cos(np.deg2rad(77.0))
TURB_MOM0_COL = "x1_mom0"
TURB_MOM2_COL = "x1_mom2"
SEC_PER_MYR = 1e6 * 3.154e7
CM_METRIC_TO_KM_MYR = SEC_PER_MYR * 1e-10
XLIM_MAIN = (1e-1, 2e3)
YLIM_MAIN = (1e-1, 2e3)
BUBBLE_MARKER_SIZE = 65
BUBBLE_MARKER = "o"
BUBBLE_BASE_FACE_COLOR = (0.5, 0.5, 0.8, 0.5)
BUBBLE_FACE_ALPHA = 0.5
BUBBLE_EDGE_ALPHA = 0.0
BUBBLE_EDGE_LW = 1.0
BUBBLE_EDGE_COLOR = (0, 0, 0, 1.0)
BUBBLE_SHOW_ERRORBAR = False
BUBBLE_ERRORBAR_USE_DATA_COLOR = True
BUBBLE_ERRORBAR_DATA_ALPHA = 0.4
BUBBLE_ERRORBAR_ELINEWIDTH = 0.6
BUBBLE_ERRORBAR_CAPSIZE = 0.0
BUBBLE_ERRORBAR_LINESTYLE = "-"
BUBBLE_ERRORBAR_ZORDER = 5

BRINKS_MARKER = "o"
BRINKS_MARKER_SIZE = 45
BRINKS_FACE_COLOR = None
BRINKS_FACE_ALPHA = 0.0
BRINKS_EDGE_COLOR = (0.8, 0.5, 0.5, 0.5)
BRINKS_EDGE_ALPHA = 0.5
BRINKS_EDGE_LW = 1.5
BRINKS_MARKER_COLOR = "0.2"
BRINKS_NHI_SCALE = 0.44
BRINKS_SHOW = True
X_LABEL_MAIN = r"$\sigma^2/\tau_{\rm diss}$ [$(\mathrm{km\,s^{-1}})^2$ Myr$^{-1}$]"
Y_LABEL_MAIN = r"$v_{\rm exp}^2/t_{\rm exp}$ [$(\mathrm{km\,s^{-1}})^2$ Myr$^{-1}$]"
LEGEND_LABEL_ONE_TO_ONE = "1:1"
LEGEND_LABEL_BRINKS = "Brinks+86"
BRINKS_INSET_LOC = "lower right"
BRINKS_INSET_SIZE = "30%"
BRINKS_INSET_BORDERPAD = 0.8
BRINKS_INSET_TEXT = "Brinks+86 (full)"
BRINKS_INSET_TEXT_POS = (0.25, 0.95)
BRINKS_INSET_TEXT_FONTSIZE = 7
BRINKS_INSET_TICK_LENGTH = 3
BRINKS_INSET_XLIM = (1e-5, 5e8)
BRINKS_INSET_YLIM = (1e-5, 5e8)
BRINKS_INSET_BBOX = (-0.0, 0.05, 1.0, 1.0)
FIGSIZE_MAIN = (4.8, 4.8)
SIGMA_LOG_SHOW = True
SIGMA_LOG_LABEL = r"$\Delta_{1:1}$"
LEGEND_BBOX = (0.05, -0.02, 1.0, 1.0)
LEGEND_LOC = "upper left"
LEGEND_FONTSIZE = 8
LEGEND_FRAME = True
LEGEND_FRAMEALPHA = 0.9
LEGEND_EDGE_COLOR = "black"
LEGEND_EDGE_LW = 0.5
LEGEND_ORDER = None  # e.g., ("bubbles", "turbulence", "brinks", "one_to_one", "sigma")
BRINKS_INSET_ENABLED = False
HIST_INSET_ENABLED = False
HIST_INSET_SIZE = "30%"
HIST_INSET_LOC = "lower right"
HIST_INSET_BORDERPAD = 0.6
HIST_INSET_BBOX = (-0.0, 0.05, 1.0, 1.0)
HIST_INSET_BINS = 20
HIST_INSET_ALPHA_BUB = 0.75
HIST_INSET_ALPHA_BRINKS = 0.75
HIST_INSET_ALPHA_TURB = 0.75
HIST_INSET_COLOR_BUB = BUBBLE_BASE_FACE_COLOR
HIST_INSET_COLOR_BRINKS = BRINKS_EDGE_COLOR
HIST_INSET_COLOR_TURB = "0.3"
HIST_INSET_TEXT = r"$n_{\rm HI}$"
HIST_INSET_TEXT_POS = (0.05, 0.95)
HIST_INSET_TEXT_FONTSIZE = 8
HIST_INSET_TICK_LENGTH = 3
HIST_INSET_LW = 1.0
HIST_INSET_LABEL_BUB = "Bubbles"
HIST_INSET_LABEL_BRINKS = "Brinks+86"
HIST_INSET_LABEL_TURB = "Turbulence"
HIST_INSET_LABEL_FONTSIZE = 7
HIST_INSET_LABEL_X = 0.03
HIST_INSET_LABEL_Y_BUB = 0.82
HIST_INSET_LABEL_Y_BRINKS = 0.7
HIST_INSET_LABEL_Y_TURB = 0.58


def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def compute_turb_sigma_and_metric(df: pd.DataFrame, branch: str):
    """
    Return r_kpc, sigma_kms^2, sigma^2/t_diss, and n_HI [cm^-3].
    t_diss = 2 h_pc / sigma; metric = sigma^3 / (2 h_pc * pc2cm) in cm^2 s^-3.
    """
    df = df.copy()
    if "r" in df.columns:
        df = df[(df["r"] >= 5000.0) & (df["r"] <= 25000.0)]
    r_pc = df["r"].to_numpy(dtype=float)
    r_kpc = r_pc / 1e3
    mom0 = df[TURB_MOM0_COL].to_numpy(dtype=float)
    mom2 = df[TURB_MOM2_COL].to_numpy(dtype=float)
    v2 = np.clip(mom2**2 - 0.6 * 8.0**2, 0.0, None)  # (km/s)^2 after correction
    sigma_kms = np.sqrt(v2)

    if branch == "max":
        h_pc = 182.0 - 37.0 + 13.0 * r_kpc
    elif branch == "min":
        h_pc = 182.0 + 37.0 + 19.0 * r_kpc
    else:
        raise ValueError("branch must be 'max' or 'min'.")

    col_den = mom0 * 1.222e6 / 1.42**2 / 3600.0 * 1.823e18
    n_hi = col_den / (2.0 * h_pc * pc2cm / COSI)

    sigma_cms = sigma_kms * 1e5
    with np.errstate(divide="ignore", invalid="ignore"):
        metric_cgs = sigma_cms**3 / (2.0 * h_pc * pc2cm)  # cm^2 s^-3
    metric_km_myr = metric_cgs * CM_METRIC_TO_KM_MYR
    return r_kpc, v2, metric_km_myr, n_hi


def interpolate_at_r(r_grid, val_grid, r_targets):
    return np.interp(r_targets, r_grid, val_grid, left=np.nan, right=np.nan)


def load_bubbles(
    bubble_table_path,
    alpha_fits,
    r_fits,
    ring_table_path=None,
    nring_column="n_HI_ring_cm-3",
):
    df_bubble = pd.read_fwf(bubble_table_path, header=0, infer_nrows=int(1e6))
    ra_pix = df_bubble["ra_pix"]
    dec_pix = df_bubble["dec_pix"]
    vexp_kms = pd.to_numeric(df_bubble["expansion_vel"], errors="coerce").to_numpy(dtype=float) + 1.0
    r_pc = pd.to_numeric(df_bubble["radius_pc"], errors="coerce").to_numpy(dtype=float)

    x_pos_pc = get_fits_at_pix(r_fits, ra_pix, dec_pix)
    x_pos_kpc = x_pos_pc / 1e3

    nring = None
    if ring_table_path is not None:
        try:
            df_ring = pd.read_fwf(ring_table_path, header=0, infer_nrows=int(1e6))
            if nring_column in df_ring.columns:
                nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)
        except Exception:
            nring = None

    vexp_cms = vexp_kms * 1e5
    with np.errstate(divide="ignore", invalid="ignore"):
        t_bub = 0.6 * r_pc * pc2cm / vexp_cms  # s
        metric_bub_cgs = vexp_cms**2 / t_bub       # cm^2 s^-3
    metric_bub_km_myr = metric_bub_cgs * CM_METRIC_TO_KM_MYR

    return {
        "R_kpc": x_pos_kpc,
        "vexp_kms": vexp_kms,
        "vexp_metric_cgs": metric_bub_km_myr,
        "nring": nring,
    }


def _auto_vmin_vmax(vals, q_lo=0.02, q_hi=0.98):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, q_lo * 100.0)
    hi = np.nanpercentile(v, q_hi * 100.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(v)
        hi = np.nanmax(v)
    return float(lo), float(hi)

def _log_limits_around(vals, pad=2.5):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return None
    vmin = np.nanmin(v) / pad
    vmax = np.nanmax(v) * pad
    if vmin <= 0:
        vmin = np.nanmin(v) / (pad * 2.0)
    return (vmin, vmax)


def _add_brinks_inset(
    ax,
    xvals,
    yvals,
    marker_size,
    marker_style,
    marker_color,
    face_color,
    face_alpha,
    edge_color,
    edge_alpha,
    marker_lw,
    inset_size=BRINKS_INSET_SIZE,
    inset_borderpad=BRINKS_INSET_BORDERPAD,
    inset_bbox=BRINKS_INSET_BBOX,
    inset_text=BRINKS_INSET_TEXT,
    inset_text_pos=BRINKS_INSET_TEXT_POS,
    inset_text_fontsize=BRINKS_INSET_TEXT_FONTSIZE,
    inset_tick_length=BRINKS_INSET_TICK_LENGTH,
    inset_loc=BRINKS_INSET_LOC,
    inset_xlim=BRINKS_INSET_XLIM,
    inset_ylim=BRINKS_INSET_YLIM,
):
    if xvals is None or yvals is None:
        return
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    mask = np.isfinite(xvals) & np.isfinite(yvals) & (xvals > 0) & (yvals > 0)
    if not np.any(mask):
        return
    x_in = xvals[mask]
    y_in = yvals[mask]
    lim_x = inset_xlim if inset_xlim is not None else _log_limits_around(x_in, pad=2.0)
    lim_y = inset_ylim if inset_ylim is not None else _log_limits_around(y_in, pad=4.0)
    if lim_x is None or lim_y is None:
        return
    face_base = marker_color if face_color is None else face_color
    face_rgba = mcolors.to_rgba(face_base, alpha=face_alpha)
    edge_base = edge_color if edge_color is not None else marker_color
    edge_rgba = mcolors.to_rgba(edge_base, alpha=edge_alpha)

    iax = inset_axes(
        ax,
        width=inset_size,
        height=inset_size,
        loc=inset_loc,
        borderpad=inset_borderpad,
        bbox_to_anchor=inset_bbox,
        bbox_transform=ax.transAxes if inset_bbox is not None else None,
    )
    iax.set_xscale("log")
    iax.set_yscale("log")
    iax.scatter(
        x_in,
        y_in,
        marker=marker_style,
        s=marker_size * 0.9,
        facecolor=face_rgba,
        edgecolor=edge_rgba,
        linewidths=marker_lw,
        zorder=6,
    )
    iax.set_xlim(lim_x)
    iax.set_ylim(lim_y)
    iax.xaxis.set_minor_locator(mticker.NullLocator())
    iax.yaxis.set_minor_locator(mticker.NullLocator())
    iax.tick_params(direction="in", which="both", length=inset_tick_length, labelsize=inset_text_fontsize, top=True, right=True)
    iax.text(
        inset_text_pos[0],
        inset_text_pos[1],
        inset_text,
        ha="left",
        va="top",
        fontsize=inset_text_fontsize,
        transform=iax.transAxes,
    )


def _add_hist_inset(
    ax,
    bubble_vals,
    brinks_vals,
    turb_vals,
    bins=HIST_INSET_BINS,
    alpha_bubble=HIST_INSET_ALPHA_BUB,
    alpha_brinks=HIST_INSET_ALPHA_BRINKS,
    alpha_turb=HIST_INSET_ALPHA_TURB,
    inset_size=HIST_INSET_SIZE,
    inset_loc=HIST_INSET_LOC,
    inset_borderpad=HIST_INSET_BORDERPAD,
    inset_bbox=HIST_INSET_BBOX,
    inset_text=HIST_INSET_TEXT,
    inset_text_pos=HIST_INSET_TEXT_POS,
    inset_text_fontsize=HIST_INSET_TEXT_FONTSIZE,
    inset_tick_length=HIST_INSET_TICK_LENGTH,
    bubble_color=HIST_INSET_COLOR_BUB,
    brinks_color=HIST_INSET_COLOR_BRINKS,
    turb_color=HIST_INSET_COLOR_TURB,
    hist_lw=HIST_INSET_LW,
    label_bub=HIST_INSET_LABEL_BUB,
    label_brinks=HIST_INSET_LABEL_BRINKS,
    label_turb=HIST_INSET_LABEL_TURB,
    label_fontsize=HIST_INSET_LABEL_FONTSIZE,
    label_x=HIST_INSET_LABEL_X,
    label_y_bub=HIST_INSET_LABEL_Y_BUB,
    label_y_brinks=HIST_INSET_LABEL_Y_BRINKS,
    label_y_turb=HIST_INSET_LABEL_Y_TURB,
):
    if bubble_vals is None and brinks_vals is None and turb_vals is None:
        return
    bvals = np.asarray(bubble_vals, dtype=float) if bubble_vals is not None else np.array([])
    bvals = bvals[np.isfinite(bvals) & (bvals > 0)]
    brvals = np.asarray(brinks_vals, dtype=float) if brinks_vals is not None else np.array([])
    brvals = brvals[np.isfinite(brvals) & (brvals > 0)]
    tvals = np.asarray(turb_vals, dtype=float) if turb_vals is not None else np.array([])
    tvals = tvals[np.isfinite(tvals) & (tvals > 0)]
    if bvals.size == 0 and brvals.size == 0 and tvals.size == 0:
        return

    combined = np.concatenate([arr for arr in (bvals, brvals, tvals) if arr.size])
    vmin = np.nanmin(combined)
    vmax = np.nanmax(combined)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return
    if isinstance(bins, int):
        bin_edges = np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)
    else:
        bin_edges = bins

    iax = inset_axes(
        ax,
        width=inset_size,
        height=inset_size,
        loc=inset_loc,
        borderpad=inset_borderpad,
        bbox_to_anchor=inset_bbox,
        bbox_transform=ax.transAxes if inset_bbox is not None else None,
    )
    iax.set_xscale("log")
    if bvals.size:
        iax.hist(
            bvals,
            bins=bin_edges,
            histtype="step",
            lw=hist_lw,
            color=mcolors.to_rgba(bubble_color, alpha=alpha_bubble),
            label="Bubbles",
        )
    if brvals.size:
        iax.hist(
            brvals,
            bins=bin_edges,
            histtype="step",
            lw=hist_lw,
            color=mcolors.to_rgba(brinks_color, alpha=alpha_brinks),
            label="Brinks+86",
        )
    if tvals.size:
        iax.hist(
            tvals,
            bins=bin_edges,
            histtype="step",
            lw=hist_lw,
            color=mcolors.to_rgba(turb_color, alpha=alpha_turb),
            label="Turbulence",
        )
    iax.tick_params(direction="in", which="both", length=inset_tick_length, labelsize=inset_text_fontsize - 1, top=True, right=True)
    iax.text(
        inset_text_pos[0],
        inset_text_pos[1],
        inset_text,
        ha="left",
        va="top",
        fontsize=inset_text_fontsize,
        transform=iax.transAxes,
    )
    if label_bub:
        iax.text(label_x, label_y_bub, label_bub, color=mcolors.to_rgba(bubble_color, alpha=1.0), fontsize=label_fontsize, ha="left", va="center", transform=iax.transAxes)
    if label_brinks:
        iax.text(label_x, label_y_brinks, label_brinks, color=mcolors.to_rgba(brinks_color, alpha=1.0), fontsize=label_fontsize, ha="left", va="center", transform=iax.transAxes)
    if label_turb:
        iax.text(label_x, label_y_turb, label_turb, color=mcolors.to_rgba(turb_color, alpha=1.0), fontsize=label_fontsize, ha="left", va="center", transform=iax.transAxes)


def plot_v2_over_t_vs_turb(
    xlim=XLIM_MAIN,
    ylim=YLIM_MAIN,
    bubble_table_path="../code/1113.tab",
    ring_table_path="../code/1113.tab",  # kept for interface symmetry; not used directly
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_value_mode="mean",  # 'mean', 'min', 'max'
    bubble_marker_size=BUBBLE_MARKER_SIZE,
    bubble_marker=BUBBLE_MARKER,
    bubble_base_face_color=BUBBLE_BASE_FACE_COLOR,
    bubble_marker_face_alpha=BUBBLE_FACE_ALPHA,
    bubble_marker_edge_alpha=BUBBLE_EDGE_ALPHA,
    bubble_marker_edge_lw=BUBBLE_EDGE_LW,
    bubble_marker_edge_color=BUBBLE_EDGE_COLOR,
    bubble_marker_zorder=6,
    bubble_show_errorbar=BUBBLE_SHOW_ERRORBAR,
    bubble_errorbar_use_data_color=BUBBLE_ERRORBAR_USE_DATA_COLOR,
    bubble_errorbar_data_alpha=BUBBLE_ERRORBAR_DATA_ALPHA,
    bubble_errorbar_color=(0, 0, 0, 0.55),
    bubble_errorbar_elinewidth=BUBBLE_ERRORBAR_ELINEWIDTH,
    bubble_errorbar_capsize=BUBBLE_ERRORBAR_CAPSIZE,
    bubble_errorbar_linestyle=BUBBLE_ERRORBAR_LINESTYLE,
    bubble_errorbar_zorder=BUBBLE_ERRORBAR_ZORDER,
    color_by_radius=False,
    cmap_name="coolwarm",
    color_vmin=None,
    color_vmax=None,
    colorbar_label=r"$R$ [kpc]",
    legend_loc=LEGEND_LOC,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_frame=LEGEND_FRAME,
    legend_framealpha=LEGEND_FRAMEALPHA,
    legend_edgecolor=LEGEND_EDGE_COLOR,
    legend_edge_lw=LEGEND_EDGE_LW,
    legend_order=LEGEND_ORDER,
    export_table=False,
    export_path="v2_over_t_vs_turb.csv",

    # Brinks+86 overlay
    brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
    brinks_show=BRINKS_SHOW,
    brinks_marker_size=BRINKS_MARKER_SIZE,
    brinks_marker=BRINKS_MARKER,
    brinks_marker_color=BRINKS_MARKER_COLOR,
    brinks_face_color=BRINKS_FACE_COLOR,
    brinks_face_alpha=BRINKS_FACE_ALPHA,
    brinks_edge_color=BRINKS_EDGE_COLOR,
    brinks_edge_alpha=BRINKS_EDGE_ALPHA,
    brinks_marker_lw=BRINKS_EDGE_LW,
    brinks_inset=BRINKS_INSET_ENABLED,
    brinks_inset_size=BRINKS_INSET_SIZE,
    brinks_inset_borderpad=BRINKS_INSET_BORDERPAD,
    brinks_inset_bbox=BRINKS_INSET_BBOX,
    brinks_inset_text=BRINKS_INSET_TEXT,
    brinks_inset_text_pos=BRINKS_INSET_TEXT_POS,
    brinks_inset_text_fontsize=BRINKS_INSET_TEXT_FONTSIZE,
    brinks_inset_tick_length=BRINKS_INSET_TICK_LENGTH,
    brinks_inset_loc=BRINKS_INSET_LOC,
    brinks_inset_xlim=BRINKS_INSET_XLIM,
    brinks_inset_ylim=BRINKS_INSET_YLIM,
    hist_inset_enabled=HIST_INSET_ENABLED,
    hist_inset_size=HIST_INSET_SIZE,
    hist_inset_loc=HIST_INSET_LOC,
    hist_inset_borderpad=HIST_INSET_BORDERPAD,
    hist_inset_bbox=HIST_INSET_BBOX,
    hist_inset_bins=HIST_INSET_BINS,
    hist_inset_alpha_bub=HIST_INSET_ALPHA_BUB,
    hist_inset_alpha_brinks=HIST_INSET_ALPHA_BRINKS,
    hist_inset_alpha_turb=HIST_INSET_ALPHA_TURB,
    hist_inset_color_turb=HIST_INSET_COLOR_TURB,
    hist_inset_color_bub=HIST_INSET_COLOR_BUB,
    hist_inset_color_brinks=HIST_INSET_COLOR_BRINKS,
    hist_inset_text=HIST_INSET_TEXT,
    hist_inset_text_pos=HIST_INSET_TEXT_POS,
    hist_inset_text_fontsize=HIST_INSET_TEXT_FONTSIZE,
    hist_inset_tick_length=HIST_INSET_TICK_LENGTH,
    hist_inset_lw=HIST_INSET_LW,
    hist_inset_label_bub=HIST_INSET_LABEL_BUB,
    hist_inset_label_brinks=HIST_INSET_LABEL_BRINKS,
    hist_inset_label_turb=HIST_INSET_LABEL_TURB,
    hist_inset_label_fontsize=HIST_INSET_LABEL_FONTSIZE,
    hist_inset_label_x=HIST_INSET_LABEL_X,
    hist_inset_label_y_bub=HIST_INSET_LABEL_Y_BUB,
    hist_inset_label_y_brinks=HIST_INSET_LABEL_Y_BRINKS,
    hist_inset_label_y_turb=HIST_INSET_LABEL_Y_TURB,

    output_pdf="v2_over_t_vs_turb.pdf",
    figsize=FIGSIZE_MAIN,
):
    """Scatter of bubble v^2/t vs turbulent v^2/t (both cm^2 s^-3), colored by R_kpc."""

    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL_MAIN)
    ax.set_ylabel(Y_LABEL_MAIN)
    ax.tick_params(direction="in", which="both", length=5, top=True, right=True)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.text(0.02, 0.98, "b", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    # turbulence curves
    df_max = pd.read_csv(prof_max_path, sep=r"\t", engine="python")
    df_min = pd.read_csv(prof_min_path, sep=r"\t", engine="python")
    r_kpc_max, _, metric_max, nhi_max = compute_turb_sigma_and_metric(df_max, branch="max")
    r_kpc_min, _, metric_min, nhi_min = compute_turb_sigma_and_metric(df_min, branch="min")

    bubble = load_bubbles(
        bubble_table_path=bubble_table_path,
        alpha_fits=alpha_fits,
        r_fits=r_fits,
        ring_table_path=ring_table_path,
        nring_column="n_HI_ring_cm-3",
    )

    turb_metric_max = interpolate_at_r(r_kpc_max, metric_max, bubble["R_kpc"])
    turb_metric_min = interpolate_at_r(r_kpc_min, metric_min, bubble["R_kpc"])
    turb_nhi_max = interpolate_at_r(r_kpc_max, nhi_max, bubble["R_kpc"])
    turb_nhi_min = interpolate_at_r(r_kpc_min, nhi_min, bubble["R_kpc"])
    if turb_value_mode == "mean":
        turb_metric = 0.5 * (turb_metric_max + turb_metric_min)
        turb_nhi = 0.5 * (turb_nhi_max + turb_nhi_min)
    elif turb_value_mode == "max":
        turb_metric = turb_metric_max
        turb_nhi = turb_nhi_max
    elif turb_value_mode == "min":
        turb_metric = turb_metric_min
        turb_nhi = turb_nhi_min
    else:
        raise ValueError("turb_value_mode must be 'mean', 'min', or 'max'.")

    r_kpc_vals = bubble["R_kpc"]
    if color_by_radius:
        cmap = plt.get_cmap(cmap_name)
        if color_vmin is None or color_vmax is None:
            vmin_auto, vmax_auto = _auto_vmin_vmax(r_kpc_vals, q_lo=0.02, q_hi=0.98)
            vmin = vmin_auto if color_vmin is None else float(color_vmin)
            vmax = vmax_auto if color_vmax is None else float(color_vmax)
        else:
            vmin = float(color_vmin)
            vmax = float(color_vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = _auto_vmin_vmax(r_kpc_vals, q_lo=0.02, q_hi=0.98)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = np.array([cmap(norm(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 0.3) for v in r_kpc_vals])
    else:
        cmap = None
        norm = None
        base_color = bubble_base_face_color if bubble_base_face_color is not None else plt.get_cmap(cmap_name)(0.6)
        base_rgba = mcolors.to_rgba(base_color, alpha=bubble_marker_face_alpha)
        colors = np.repeat(np.array(base_rgba)[None, :], len(r_kpc_vals), axis=0)

    facecolors = np.stack([np.r_[c[:3], bubble_marker_face_alpha] for c in colors])
    edge_rgba = mcolors.to_rgba(bubble_marker_edge_color, alpha=bubble_marker_edge_alpha)
    edgecolors = np.repeat(np.array(edge_rgba)[None, :], len(colors), axis=0)

    mask = np.isfinite(turb_metric) & np.isfinite(bubble["vexp_metric_cgs"])
    ax.scatter(
        turb_metric[mask],
        bubble["vexp_metric_cgs"][mask],
        s=bubble_marker_size,
        marker=bubble_marker,
        facecolor=facecolors[mask],
        edgecolor=edgecolors[mask],
        linewidths=bubble_marker_edge_lw,
        zorder=bubble_marker_zorder,
    )

    if bubble_show_errorbar:
        xerr = 0.05 * turb_metric
        yerr = 0.5 * bubble["vexp_metric_cgs"]
        if bubble_errorbar_use_data_color and color_by_radius and cmap is not None and norm is not None:
            for xi, yi, xe, ye, col in zip(turb_metric[mask], bubble["vexp_metric_cgs"][mask], xerr[mask], yerr[mask], colors[mask]):
                ecolor = (col[0], col[1], col[2], float(bubble_errorbar_data_alpha))
                ax.errorbar(
                    [xi], [yi],
                    xerr=[[xe], [xe]], yerr=[[ye], [ye]],
                    fmt='none',
                    ecolor=ecolor,
                    elinewidth=bubble_errorbar_elinewidth,
                    capsize=bubble_errorbar_capsize,
                    ls=bubble_errorbar_linestyle,
                    zorder=bubble_errorbar_zorder
                )
        else:
            ax.errorbar(
                turb_metric,
                bubble["vexp_metric_cgs"],
                xerr=xerr,
                yerr=yerr,
                fmt='none',
                ecolor=bubble_errorbar_color,
                elinewidth=bubble_errorbar_elinewidth,
                capsize=bubble_errorbar_capsize,
                ls=bubble_errorbar_linestyle,
                zorder=bubble_errorbar_zorder
            )

    if xlim is not None and ylim is not None:
        line_lo = max(xlim[0], ylim[0])
        line_hi = min(xlim[1], ylim[1])
        line_x = np.array([line_lo, line_hi])
        ax.plot(line_x, line_x, color="k", lw=0.8, ls="--", zorder=4, label=LEGEND_LABEL_ONE_TO_ONE)

    def _sigma_log(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if not np.any(m):
            return np.nan
        d = np.log10(y[m] / x[m])
        return np.sqrt(np.nanmean(d**2))

    sigma_log = _sigma_log(turb_metric[mask], bubble["vexp_metric_cgs"][mask])
    leg_sigma = None
    if np.isfinite(sigma_log):
        leg_sigma = Line2D([], [], marker="None", linestyle="None", label=rf"{SIGMA_LOG_LABEL} = {sigma_log:.2f} dex")

    # Brinks+86 overlay as crosses
    brinks_handle = None
    brinks_x = None
    brinks_y = None
    brinks_nhi = None
    if brinks_show and brinks_table_path is not None:
        bpath = Path(brinks_table_path)
        if bpath.exists():
            bdf = pd.read_fwf(bpath)
            required = ["R_kpc", "DV_kms", "Diam_pc", "Mass_1e4Msun"]
            missing = [c for c in required if c not in bdf.columns]
            if not missing:
                r_b = pd.to_numeric(bdf["R_kpc"], errors="coerce").to_numpy(dtype=float)
                dv_b = np.abs(pd.to_numeric(bdf["DV_kms"], errors="coerce").to_numpy(dtype=float))
                diam_pc = pd.to_numeric(bdf["Diam_pc"], errors="coerce").to_numpy(dtype=float)
                if "nHI_cm3" in bdf.columns:
                    brinks_nhi = pd.to_numeric(bdf["nHI_cm3"], errors="coerce").to_numpy(dtype=float) * BRINKS_NHI_SCALE
                mass_1e4 = pd.to_numeric(bdf["Mass_1e4Msun"], errors="coerce").to_numpy(dtype=float)
                radius_pc = 0.5 * diam_pc
                vexp_cms_b = dv_b * 1e5
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_b = 0.6 * radius_pc * pc2cm / vexp_cms_b
                    metric_b_cgs = vexp_cms_b**2 / t_b
                metric_b = metric_b_cgs * CM_METRIC_TO_KM_MYR

                turb_b_max = interpolate_at_r(r_kpc_max, metric_max, r_b)
                turb_b_min = interpolate_at_r(r_kpc_min, metric_min, r_b)
                if turb_value_mode == "mean":
                    turb_both = 0.5 * (turb_b_max + turb_b_min)
                elif turb_value_mode == "max":
                    turb_both = turb_b_max
                else:
                    turb_both = turb_b_min

                mb = np.isfinite(turb_both) & np.isfinite(metric_b)
                brinks_x = turb_both
                brinks_y = metric_b
                if np.any(mb):
                    brinks_handle = ax.scatter(
                        turb_both[mb],
                        metric_b[mb],
                        marker=brinks_marker,
                        s=brinks_marker_size,
                        facecolor=mcolors.to_rgba(
                            brinks_marker_color if brinks_face_color is None else brinks_face_color,
                            alpha=brinks_face_alpha,
                        ),
                        edgecolor=mcolors.to_rgba(
                            brinks_edge_color if brinks_edge_color is not None else brinks_marker_color,
                            alpha=brinks_edge_alpha,
                        ),
                        linewidths=brinks_marker_lw,
                        zorder=6,
                        label=LEGEND_LABEL_BRINKS,
                    )
        else:
            print(f"[V2OverTvsTurb] Brinks table not found at {bpath}, skipping crosses.")

    if color_by_radius and cmap is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)
        cbar.set_label(colorbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)

    if brinks_inset:
        inset_xlim_auto = None
        inset_ylim_auto = None
        if brinks_inset_xlim is None and brinks_x is not None:
            inset_xlim_auto = _log_limits_around(brinks_x, pad=5.0)
        if brinks_inset_ylim is None and brinks_y is not None:
            inset_ylim_auto = _log_limits_around(brinks_y, pad=5.0)
        _add_brinks_inset(
            ax=ax,
            xvals=brinks_x,
            yvals=brinks_y,
            marker_size=brinks_marker_size,
            marker_style=brinks_marker,
            marker_color=brinks_marker_color,
            face_color=brinks_face_color,
            face_alpha=brinks_face_alpha,
            edge_color=brinks_edge_color,
            edge_alpha=brinks_edge_alpha,
            marker_lw=brinks_marker_lw,
            inset_size=brinks_inset_size,
            inset_borderpad=brinks_inset_borderpad,
            inset_bbox=brinks_inset_bbox,
            inset_text=brinks_inset_text,
            inset_text_pos=brinks_inset_text_pos,
            inset_text_fontsize=brinks_inset_text_fontsize,
            inset_tick_length=brinks_inset_tick_length,
            inset_loc=brinks_inset_loc,
            inset_xlim=inset_xlim_auto if brinks_inset_xlim is None else brinks_inset_xlim,
            inset_ylim=inset_ylim_auto if brinks_inset_ylim is None else brinks_inset_ylim,
        )

    if hist_inset_enabled:
        _add_hist_inset(
            ax=ax,
            bubble_vals=bubble.get("nring"),
            brinks_vals=brinks_nhi,
            turb_vals=turb_nhi,
            bins=hist_inset_bins,
            alpha_bubble=hist_inset_alpha_bub,
            alpha_brinks=hist_inset_alpha_brinks,
            alpha_turb=hist_inset_alpha_turb,
            inset_size=hist_inset_size,
            inset_loc=hist_inset_loc,
            inset_borderpad=hist_inset_borderpad,
            inset_bbox=hist_inset_bbox,
            inset_text=hist_inset_text,
            inset_text_pos=hist_inset_text_pos,
            inset_text_fontsize=hist_inset_text_fontsize,
            inset_tick_length=hist_inset_tick_length,
            bubble_color=hist_inset_color_bub,
            brinks_color=hist_inset_color_brinks,
            turb_color=hist_inset_color_turb,
            hist_lw=hist_inset_lw,
            label_bub=hist_inset_label_bub,
            label_brinks=hist_inset_label_brinks,
            label_turb=hist_inset_label_turb,
            label_fontsize=hist_inset_label_fontsize,
            label_x=hist_inset_label_x,
            label_y_bub=hist_inset_label_y_bub,
            label_y_brinks=hist_inset_label_y_brinks,
            label_y_turb=hist_inset_label_y_turb,
        )

    handles, labels = ax.get_legend_handles_labels()
    if leg_sigma is not None:
        handles.append(leg_sigma)
        labels.append(leg_sigma.get_label())
    if legend_order:
        order_map = {lbl: hdl for hdl, lbl in zip(handles, labels)}
        ordered = []
        for key in legend_order:
            if key in order_map:
                ordered.append((order_map[key], key))
        for hdl, lbl in zip(handles, labels):
            if lbl not in [l for _, l in ordered]:
                ordered.append((hdl, lbl))
        handles = [h for h, _ in ordered]
        labels = [l for _, l in ordered]

    leg = ax.legend(
        handles,
        labels,
        loc=legend_loc,
        fontsize=legend_fontsize,
        frameon=legend_frame,
        framealpha=legend_framealpha,
        edgecolor=legend_edgecolor,
        bbox_to_anchor=LEGEND_BBOX,
    )
    if legend_frame:
        leg.get_frame().set_linewidth(legend_edge_lw)

    if export_table:
        export_df = pd.DataFrame({
            "R_kpc": bubble["R_kpc"],
            "turb_v2_over_t_cgs": turb_metric,
            "bubble_v2_over_t_cgs": bubble["vexp_metric_cgs"],
        })
        export_path = Path(export_path)
        export_df.to_csv(export_path, index=False)
        print(f"[V2OverTvsTurb] Exported table to {export_path}")

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    plot_v2_over_t_vs_turb()
