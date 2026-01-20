#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy budget figure — turbulence band + bubbles + optional MRI curve (continuous n_ring coloring)

This version uses:
- nring from '../code/1110.tab' : column 'n_HI_ring_cm-3' for BOTH dotE computation and bubble COLORING.
- Coloring is continuous (no discrete bins). The color scale uses a linear Normalize
  with vmin/vmax taken automatically from data percentiles unless you override.
- MRI curve derived from the median resampled profile (profile_resampled_med.tsv) following main_chart_sum.

Output:
- dot_e_balance.pdf (or the name you pass in)
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

# -----------------------
# Constants and column choices
# -----------------------
pc2cm = 3e18
SEC_PER_MYR = 1e6 * 3.154e7
COSI = np.cos(np.deg2rad(77.0))

FIG_DPI = 100
SAVEFIG_DPI = 100
AX_LABELSIZE = 10
AX_TITLESIZE = 10
TICK_LABELSIZE = 10
TICK_LENGTH_MAIN = 5
COLORBAR_PAD = 0.01
COLORBAR_ASPECT = 40
COLORBAR_LABELSIZE = 9
COLORBAR_TICKSIZE = 8
LEGEND_LOC = "lower left"
LEGEND_FONTSIZE = 7
LEGEND_FRAMEALPHA = 0.9
LEGEND_EDGE_COLOR = "black"
LEGEND_EDGE_LW = 0.5
LEGEND_ORDER = ("bubbles", "brinks", "turbulence", "mri")
FIGSIZE_MAIN = (4.8, 4.8)

BUBBLE_MARKER_SIZE = 65
BUBBLE_MARKER = "o"
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)
BUBBLE_FACE_ALPHA = 0.7
BUBBLE_EDGE_ALPHA = 0.0
BUBBLE_EDGE_LW = 0.0
BUBBLE_EDGE_COLOR = (0.9, 0.2, 0.2)

BRINKS_MARKER = "^"
BRINKS_MARKER_SIZE = 55
BRINKS_FACE_COLOR = None
BRINKS_FACE_ALPHA = 0.0
BRINKS_EDGE_COLOR = (0.6, 0.6, 0.9)
BRINKS_EDGE_ALPHA = 0.6
BRINKS_EDGE_LW = 0
BRINKS_MARKER_COLOR = (0.6, 0.6, 0.9)
BRINKS_ENERGY_SCALE = 0.44
BRINKS_INSET_XLIM = (0,25)
BRINKS_INSET_YLIM = (1e-3,5e6)
BRINKS_INSET_WIDTH = "33%"
BRINKS_INSET_HEIGHT = "33%"
BRINKS_INSET_LOC = "upper right"
BRINKS_INSET_BORDERPAD = 0.8
BRINKS_INSET_TEXT = "Brinks+86 (full)"
BRINKS_INSET_TEXT_POS = (0.05, 0.95)
BRINKS_INSET_TEXT_FONTSIZE = 7
BRINKS_INSET_TICK_LENGTH = 3
X_LABEL_MAIN = "Distance to M31's center: $R$ [kpc]"
Y_LABEL_MAIN = r"$\dot{e}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]"
LEGEND_LABEL_BUBBLES = "New Bubbles"
LEGEND_LABEL_TURB = "Turbulence"
LEGEND_LABEL_BRINKS = "Brinks+86"
BUBBLE_SHOW_ERRORBAR = True
BUBBLE_ERRORBAR_USE_DATA_COLOR = True
BUBBLE_ERRORBAR_DATA_ALPHA = 0.4
BUBBLE_ERRORBAR_ELINEWIDTH = 0.8
BUBBLE_ERRORBAR_CAPSIZE = 0.0
BUBBLE_ERRORBAR_LINESTYLE = "-"
BUBBLE_ERRORBAR_ZORDER = 5

# Which profile columns to use for turbulence computation:
TURB_MOM0_COL = "x1_mom0"
TURB_MOM2_COL = "x1_mom2"


# -----------------------
# Utilities
# -----------------------
def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def clip_xy(x, y, xlo, xhi):
    m = np.isfinite(x) & np.isfinite(y) & (x >= xlo) & (x <= xhi)
    if not np.any(m):
        return np.array([]), np.array([])
    xs = np.asarray(x)[m]
    ys = np.asarray(y)[m]
    order = np.argsort(xs)
    return xs[order], ys[order]


def compute_turb_from_profile_df(df: pd.DataFrame, branch: str):
    """
    Compute per-radius turbulent power density from a resampled profile DataFrame.

    df columns: ['r', TURB_MOM0_COL, TURB_MOM2_COL], with 'r' in pc.
    branch: 'max' -> h_pc = 182 - 37 + 13 * r_kpc
            'min' -> h_pc = 182 + 37 + 19 * r_kpc
    return: r_kpc, rate  (10^-28 erg cm^-3 s^-1)
    """
    r_pc = df["r"].to_numpy(dtype=float)
    r_kpc = r_pc / 1e3
    mom0 = df[TURB_MOM0_COL].to_numpy(dtype=float)
    mom2 = df[TURB_MOM2_COL].to_numpy(dtype=float)

    if branch == "max":
        h_pc = 182.0 - 37.0 + 13.0 * r_kpc
    elif branch == "min":
        h_pc = 182.0 + 37.0 + 19.0 * r_kpc
    else:
        raise ValueError("branch must be 'max' or 'min'.")

    # column density [cm^-2]
    col_den = mom0 * 1.222e6 / 1.42**2 / 3600.0 * 1.823e18
    # number density [cm^-3]
    n_HI = col_den / (2.0 * h_pc * pc2cm / COSI)

    # velocity dispersion (km/s -> cm/s), corrected
    v2 = np.clip(mom2**2 - 0.6 * 8.0**2, 0.0, None)
    v = np.sqrt(v2) * 1e5

    # dissipation time ~ crossing time
    t_diss = np.where(v > 0.0, 2.0 * h_pc * pc2cm / v, np.inf)

    rho = 1.67e-24 * n_HI
    ek_density = 0.5 * rho * v**2
    rate = ek_density / t_diss * 1e28
    return r_kpc, rate


def compute_mri_from_resampled(df_med: pd.DataFrame):
    """
    Compute MRI heating curve from a resampled profile DataFrame (median branch).

    df_med columns: ['r', TURB_MOM0_COL] with 'r' in pc.
    returns: r_kpc, rate (10^-28 erg cm^-3 s^-1)
    """
    r_pc = df_med["r"].to_numpy(dtype=float)
    r_kpc = r_pc / 1e3
    mom0 = df_med[TURB_MOM0_COL].to_numpy(dtype=float)

    H_pc = 182 + 16 * r_kpc
    col_den = mom0 * 1.222e6 / 1.42 / 3600 * 1.823e18
    num_den = col_den / (2 * H_pc * pc2cm / COSI)
    rho = num_den * 1.673e-24

    R_cm = r_kpc * 3.086e21
    H_cm = H_pc * pc2cm
    Omega = 2.2e7 / R_cm
    vA_max = np.sqrt(2) * H_cm * Omega
    B_max = vA_max * np.sqrt(rho)
    B_norm, Omega_norm = 3e-6, 1 / (220e6 * 3.154e7)
    y_mri = 3e-29 * (B_max / B_norm)**2 * (Omega / Omega_norm) * 1e28 * 3

    return r_kpc, y_mri


def draw_turb_band(
    ax,
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_xlim=(5.0, 24.0),
    # style
    color=(0.6, 0.6, 0.6, 0.35),
    edgecolor=(0.1, 0.1, 0.1, 0.85),
    linewidth=0.4,
    zorder=-1000,
    label=r"$\dot{e}_{\rm turb}$",
    # union or intersect
    turb_join="intersect",
    extrapolate_union=True,
):
    df_max = pd.read_csv(prof_max_path, sep=r"\t", engine="python")
    df_min = pd.read_csv(prof_min_path, sep=r"\t", engine="python")

    r_kpc_max, rate_max = compute_turb_from_profile_df(df_max, branch="max")
    r_kpc_min, rate_min = compute_turb_from_profile_df(df_min, branch="min")

    xlo_t, xhi_t = turb_xlim
    x_max, y_max = clip_xy(r_kpc_max, rate_max, xlo_t, xhi_t)
    x_min, y_min = clip_xy(r_kpc_min, rate_min, xlo_t, xhi_t)

    if x_max.size == 0 or x_min.size == 0:
        return None

    if turb_join == "intersect":
        xl = max(x_max.min(), x_min.min())
        xr = min(x_max.max(), x_min.max())
        if xl >= xr:
            return None
        ref_x = x_max if x_max.size >= x_min.size else x_min
        ref_x = ref_x[(ref_x >= xl) & (ref_x <= xr)]
        y_up = np.interp(ref_x, x_max, y_max)
        y_lo = np.interp(ref_x, x_min, y_min)
        mm = np.isfinite(y_up) & np.isfinite(y_lo)
        if not np.any(mm):
            return None
        return ax.fill_between(
            ref_x[mm], y_lo[mm], y_up[mm],
            color=color, edgecolor=edgecolor, linewidth=linewidth,
            label=label, zorder=zorder
        )

    # union with optional endpoint linear extrapolation
    x_comb = np.unique(np.concatenate([x_min, x_max]))

    def interp_with_optional_extrap(x_src, y_src, x_ref):
        y_ref = np.interp(x_ref, x_src, y_src, left=np.nan, right=np.nan)
        if extrapolate_union and x_src.size >= 2:
            m_left = x_ref < x_src[0]
            if np.any(m_left):
                aL = (y_src[1] - y_src[0]) / (x_src[1] - x_src[0])
                bL = y_src[0] - aL * x_src[0]
                y_ref[m_left] = aL * x_ref[m_left] + bL
            m_right = x_ref > x_src[-1]
            if np.any(m_right):
                aR = (y_src[-1] - y_src[-2]) / (x_src[-1] - x_src[-2])
                bR = y_src[-1] - aR * x_src[-1]
                y_ref[m_right] = aR * x_ref[m_right] + bR
        return y_ref

    y_up = interp_with_optional_extrap(x_max, y_max, x_comb)
    y_lo = interp_with_optional_extrap(x_min, y_min, x_comb)
    mm = np.isfinite(y_up) & np.isfinite(y_lo)
    if not np.any(mm):
        return None
    return ax.fill_between(
        x_comb[mm], y_lo[mm], y_up[mm],
        color=color, edgecolor=edgecolor, linewidth=linewidth,
        label=label, zorder=zorder
    )


def _auto_vmin_vmax(vals, q_lo=0.02, q_hi=0.98):
    """Robust vmin/vmax from percentiles; falls back to finite min/max."""
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
    brinks_inset_xlim=None,
    brinks_inset_ylim=None,
    inset_width=BRINKS_INSET_WIDTH,
    inset_height=BRINKS_INSET_HEIGHT,
    inset_loc=BRINKS_INSET_LOC,
    inset_borderpad=BRINKS_INSET_BORDERPAD,
    inset_text=BRINKS_INSET_TEXT,
    inset_text_pos=BRINKS_INSET_TEXT_POS,
    inset_text_fontsize=BRINKS_INSET_TEXT_FONTSIZE,
    tick_length=BRINKS_INSET_TICK_LENGTH,
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
    lim_x = brinks_inset_xlim if brinks_inset_xlim is not None else _log_limits_around(x_in, pad=2.0)
    lim_y = brinks_inset_ylim if brinks_inset_ylim is not None else _log_limits_around(y_in, pad=4.0)
    if lim_x is None or lim_y is None:
        return
    face_base = marker_color if face_color is None else face_color
    face_rgba = "none" if float(face_alpha) == 0 else mcolors.to_rgba(face_base, alpha=face_alpha)
    edge_base = edge_color if edge_color is not None else marker_color
    edge_rgba = mcolors.to_rgba(edge_base, alpha=edge_alpha)

    iax = inset_axes(ax, width=inset_width, height=inset_height, loc=inset_loc, borderpad=inset_borderpad)
    iax.set_xscale("linear")
    iax.set_yscale("log")
    iax.scatter(
        x_in,
        y_in,
        marker=marker_style,
        s=marker_size * 0.9,
        facecolor=edge_rgba,
        edgecolor=edge_rgba,
        linewidths=marker_lw,
        zorder=12,
    )
    iax.set_xlim(lim_x)
    iax.set_ylim(lim_y)
    iax.yaxis.set_minor_locator(mticker.NullLocator())
    iax.tick_params(direction="in", which="both", length=tick_length, labelsize=inset_text_fontsize, top=True, right=True)
    iax.text(
        inset_text_pos[0],
        inset_text_pos[1],
        inset_text,
        ha="left",
        va="top",
        fontsize=inset_text_fontsize,
        transform=iax.transAxes,
    )


# -----------------------
# Main plotting function
# -----------------------
def plot_energy_budget_no_mri(
    # axes limits
    xlim_full=(4.0, 28.0),
    ylim=(2e-2, 2e2),  # y-axis range is user adjustable

    # turbulence band input and style
    turb_xlim=(5.0, 24.0),
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_color=(0.6, 0.6, 0.6, 0.35),
    turb_edgecolor=(0.1, 0.1, 0.1, 0.85),
    turb_linewidth=0.4,
    turb_zorder=-1000,
    turb_label=r"$\dot{e}_{\rm turb}$",
    turb_join="intersect",
    turb_extrapolate_union=True,

    # MRI (from resampled_med profile)
    mri_show=True,
    mri_profile_path="../data/profile_resampled_med.tsv",
    mri_xlim=(4.0, 28.0),
    mri_mode="shade",           # 'shade' or 'line'
    mri_shade_color=(0.2, 0.9, 0.2, 0.26),
    mri_line_color=(0.3, 0.3, 0.3, 0.95),
    mri_linewidth=0.6,
    mri_linestyle="--",
    mri_label=r"$\mathrm{MRI}$",

    # bubbles input
    bubble_table_path="../code/1113.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",

    # ring table and columns
    ring_table_path="../code/1110.tab",
    nring_column="n_HI_ring_cm-3",

    # continuous color map settings for n_ring
    color_by_value=False,
    color_cmap_name="coolwarm",
    color_vmin=None,   # if None, auto from data percentiles
    color_vmax=None,   # if None, auto from data percentiles
    colorbar_label=r"$n_{\rm HI}$ in ring [cm$^{-3}$]",
    colorbar_aspect=COLORBAR_ASPECT,

    # Brinks+86 comparison (grey dots)
    brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
    brinks_show=True,
    brinks_marker_size=BRINKS_MARKER_SIZE,
    brinks_marker=BRINKS_MARKER,
    brinks_marker_color=BRINKS_MARKER_COLOR,
    brinks_face_color=BRINKS_FACE_COLOR,
    brinks_face_alpha=BRINKS_FACE_ALPHA,
    brinks_edge_color=BRINKS_EDGE_COLOR,
    brinks_edge_alpha=BRINKS_EDGE_ALPHA,
    brinks_marker_edge=BRINKS_EDGE_COLOR,
    brinks_marker_lw=BRINKS_EDGE_LW,
    brinks_inset=True,
    brinks_inset_width=BRINKS_INSET_WIDTH,
    brinks_inset_height=BRINKS_INSET_HEIGHT,
    brinks_inset_loc=BRINKS_INSET_LOC,
    brinks_inset_borderpad=BRINKS_INSET_BORDERPAD,
    brinks_inset_text=BRINKS_INSET_TEXT,
    brinks_inset_text_pos=BRINKS_INSET_TEXT_POS,
    brinks_inset_text_fontsize=BRINKS_INSET_TEXT_FONTSIZE,
    brinks_inset_tick_length=BRINKS_INSET_TICK_LENGTH,
    brinks_inset_xlim=BRINKS_INSET_XLIM,
    brinks_inset_ylim=BRINKS_INSET_YLIM,
    brinks_export_table=True,
    brinks_export_path=None,

    # bubble drawing and styles
    bubble_mode="circle",   # 'rect' or 'circle'
    # rectangle styles
    bubble_rect_face_alpha=0.15,
    bubble_rect_edge_alpha=0.95,
    bubble_rect_edge_lw=0.0,
    bubble_rect_zorder=5,
    # circle styles
    bubble_marker_size=BUBBLE_MARKER_SIZE,
    bubble_marker=BUBBLE_MARKER,
    bubble_base_face_color=BUBBLE_BASE_FACE_COLOR,
    bubble_marker_face_alpha=BUBBLE_FACE_ALPHA,
    bubble_use_data_edgecolor=False,
    bubble_marker_edge_color=BUBBLE_EDGE_COLOR,
    bubble_marker_edge_alpha=BUBBLE_EDGE_ALPHA,
    bubble_marker_edge_lw=BUBBLE_EDGE_LW,
    bubble_marker_zorder=6,
    # errorbar control for circle mode
    bubble_show_errorbar=BUBBLE_SHOW_ERRORBAR,
    bubble_errorbar_use_data_color=BUBBLE_ERRORBAR_USE_DATA_COLOR,
    bubble_errorbar_data_alpha=BUBBLE_ERRORBAR_DATA_ALPHA,
    bubble_errorbar_color=(0, 0, 0, 0.55),
    bubble_errorbar_elinewidth=BUBBLE_ERRORBAR_ELINEWIDTH,
    bubble_errorbar_capsize=BUBBLE_ERRORBAR_CAPSIZE,
    bubble_errorbar_linestyle=BUBBLE_ERRORBAR_LINESTYLE,
    bubble_errorbar_zorder=BUBBLE_ERRORBAR_ZORDER,

    # legend
    show_legend=True,
    legend_loc=LEGEND_LOC,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_frame=True,
    legend_framealpha=LEGEND_FRAMEALPHA,
    legend_edgecolor=LEGEND_EDGE_COLOR,
    legend_edge_lw=LEGEND_EDGE_LW,
    legend_order=LEGEND_ORDER,

    # output
    output_pdf="dot_e_balance.pdf",
    # figure and ticks
    figsize=FIGSIZE_MAIN,
    fig_dpi=FIG_DPI,
    savefig_dpi=SAVEFIG_DPI,
    axes_labelsize=AX_LABELSIZE,
    axes_titlesize=AX_TITLESIZE,
    tick_labelsize=TICK_LABELSIZE,
    tick_length_main=TICK_LENGTH_MAIN,
):
    """
    Turbulence band from MAX and MIN resampled profiles.
    Optional MRI curve from median resampled profile.
    Bubbles are colored continuously by n_ring using a linear Normalize.
    n_ring from ring_table_path also enters dotE.
    """

    # base style
    plt.rcParams.update({
        "figure.dpi": fig_dpi,
        "savefig.dpi": savefig_dpi,
        "axes.labelsize": axes_labelsize,
        "axes.titlesize": axes_titlesize,
        "xtick.labelsize": tick_labelsize,
        "ytick.labelsize": tick_labelsize
    })
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL_MAIN)
    ax.set_ylabel(Y_LABEL_MAIN)
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.tick_params(direction="in", which="both", length=tick_length_main, top=True, bottom=True, left=True, right=True)
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Turbulence band
    turb_band = draw_turb_band(
        ax,
        prof_max_path=prof_max_path,
        prof_min_path=prof_min_path,
        turb_xlim=turb_xlim,
        color=turb_color,
        edgecolor=turb_edgecolor,
        linewidth=turb_linewidth,
        zorder=turb_zorder,
        label=turb_label,
        turb_join=turb_join,
        extrapolate_union=turb_extrapolate_union,
    )

    # Bubbles data preparation
    # 1) geometry from bubble table and FITS
    df_bubble = pd.read_fwf(bubble_table_path, header=0, infer_nrows=int(1e6))
    ra_pix = df_bubble["ra_pix"]
    dec_pix = df_bubble["dec_pix"]
    r_bub_pc = df_bubble["radius_pc"].values
    vexp_kms = df_bubble["expansion_vel"].values + 1  # small offset, consistent with earlier usage

    angles = get_fits_at_pix(alpha_fits, ra_pix, dec_pix) - 90.0
    x_pos = get_fits_at_pix(r_fits, ra_pix, dec_pix) / 1e3  # kpc

    # 2) read nring from 1110.tab
    df_ring = pd.read_fwf(ring_table_path, header=0, infer_nrows=int(1e6))
    if nring_column not in df_ring.columns:
        raise KeyError(f"'{nring_column}' not found in {ring_table_path}")

    nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)

    # 3) require same length for one to one alignment
    if len(df_ring) != len(df_bubble):
        raise ValueError(
            f"Row count mismatch: ring table ({len(df_ring)}) vs bubble table ({len(df_bubble)}). "
            f"Please ensure the two tables are aligned row by row or add a common 'id' to join."
        )

    # 4) use nring to compute dotE
    h_pc = 182 + 16 * x_pos  # scale height
    rho = 1.673e-24 * nring  # mass density
    v_cms = (vexp_kms * 1e5)
    dotE = 0.5 * rho * v_cms**3 / (h_pc * pc2cm) * 1e28

    # print min and max
    def _print_minmax(name, arr):
        arr = np.asarray(arr, dtype=float)
        m = np.isfinite(arr)
        if not np.any(m):
            print(f"[EnergyBudget] {name}: no finite values")
            return
        print(f"[EnergyBudget] {name}: min={np.nanmin(arr[m]):.6g}, max={np.nanmax(arr[m]):.6g}, N={np.count_nonzero(m)}")

    _print_minmax("R_kpc (x_pos)", x_pos)
    _print_minmax("dotE (y)", dotE)
    _print_minmax("n_ring [cm^-3]", nring)

    # continuous color mapping by nring
    if color_by_value:
        if color_vmin is None or color_vmax is None:
            vmin_auto, vmax_auto = _auto_vmin_vmax(nring, q_lo=0.02, q_hi=0.98)
            vmin = vmin_auto if color_vmin is None else float(color_vmin)
            vmax = vmax_auto if color_vmax is None else float(color_vmax)
        else:
            vmin = float(color_vmin)
            vmax = float(color_vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = _auto_vmin_vmax(nring, q_lo=0.02, q_hi=0.98)
        cmap = plt.get_cmap(color_cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax)
        nring_colors = np.array([cmap(norm(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 0.3) for v in nring])
    else:
        cmap = None
        norm = None
        base_color_input = bubble_base_face_color if bubble_base_face_color is not None else plt.get_cmap(color_cmap_name)(0.6)
        base_rgba = mcolors.to_rgba(base_color_input, alpha=bubble_marker_face_alpha)
        nring_colors = np.repeat(np.array(base_rgba)[None, :], len(nring), axis=0)

    # Draw bubbles
    if bubble_mode == "rect":
        for i in range(len(x_pos)):
            h_i = 182 + 16 * x_pos[i]
            dh = h_i * np.tan(np.deg2rad(77)) * np.cos(np.deg2rad(angles[i]))
            dr = r_bub_pc[i]
            dxk = np.sqrt(dh**2 + dr**2) / 1e3
            if color_by_value and cmap is not None and norm is not None:
                base_color = cmap(norm(nring[i])) if np.isfinite(nring[i]) else (0.3, 0.3, 0.3, 0.2)
                fc = base_color[:3] + (bubble_rect_face_alpha,)
                ec = base_color[:3] + (bubble_rect_edge_alpha,)
            else:
                base_color = nring_colors[i]
                fc = base_color[:3] + (bubble_rect_face_alpha,)
                ec = base_color[:3] + (bubble_rect_edge_alpha,)
            rect = patches.Rectangle(
                (x_pos[i] - dxk, dotE[i] * 0.5),
                2 * dxk, dotE[i],
                facecolor=fc,
                edgecolor=ec,
                lw=bubble_rect_edge_lw,
                zorder=bubble_rect_zorder
            )
            ax.add_patch(rect)

    elif bubble_mode == "circle":
        data_colors = nring_colors
        facecolors = np.stack([np.r_[c[:3], bubble_marker_face_alpha] for c in data_colors])

        if bubble_use_data_edgecolor and color_by_value and cmap is not None and norm is not None:
            edgecolors = np.stack([np.r_[c[:3], bubble_marker_edge_alpha] for c in data_colors])
        else:
            edge_rgba = mcolors.to_rgba(bubble_marker_edge_color, alpha=bubble_marker_edge_alpha)
            edgecolors = np.repeat(np.array(edge_rgba)[None, :], len(x_pos), axis=0)

        ax.scatter(
            x_pos, dotE,
            s=bubble_marker_size,
            marker=bubble_marker,
            facecolor=facecolors,
            edgecolor=edgecolors,
            linewidths=bubble_marker_edge_lw,
            zorder=bubble_marker_zorder
        )

        # error bars
        if bubble_show_errorbar:
            h_i = 182 + 16 * x_pos
            dh = h_i * np.tan(np.deg2rad(77)) * np.cos(np.deg2rad(angles))
            dr = r_bub_pc
            dxk = np.sqrt(dh**2 + dr**2) / 1e3
            xerr = dxk
            yerr = 0.5 * dotE

            if bubble_errorbar_use_data_color:
                for xi, yi, xe, ye, col in zip(x_pos, dotE, xerr, yerr, data_colors):
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
                    x_pos, dotE,
                    xerr=xerr, yerr=yerr,
                    fmt='none',
                    ecolor=bubble_errorbar_color,
                    elinewidth=bubble_errorbar_elinewidth,
                    capsize=bubble_errorbar_capsize,
                    ls=bubble_errorbar_linestyle,
                    zorder=bubble_errorbar_zorder
                )
    else:
        raise ValueError("bubble_mode must be 'rect' or 'circle'.")

    # colorbar for continuous nring coloring
    if color_by_value and cmap is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=COLORBAR_PAD, aspect=colorbar_aspect)
        cbar.set_label(colorbar_label, fontsize=COLORBAR_LABELSIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    # set axes limits before adding theory curves
    ax.set_xlim(xlim_full)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)

    # MRI curve
    mri_handle = None
    if mri_show and mri_profile_path is not None:
        try:
            df_med_mri = pd.read_csv(mri_profile_path, sep=r"\t", engine="python")
            x_mri_all, y_mri_all = compute_mri_from_resampled(df_med_mri)
            x_mri, y_mri = clip_xy(x_mri_all, y_mri_all, mri_xlim[0], mri_xlim[1])
            if x_mri.size:
                if mri_mode == "shade":
                    ybase = ax.get_ylim()[0]
                    mri_handle = ax.fill_between(x_mri, ybase, y_mri, color=mri_shade_color, linewidth=0.0, zorder = 4)
                elif mri_mode == "line":
                    (mri_handle,) = ax.plot(
                        x_mri,
                        y_mri,
                        color=mri_line_color,
                        lw=mri_linewidth,
                        ls=mri_linestyle,
                    )
                else:
                    raise ValueError("mri_mode must be 'shade' or 'line'.")
        except Exception as exc:
            print(f"[EnergyBudget] Warning: failed to load/plot MRI curve: {exc}")

    brinks_handle = None
    brinks_age_myr = None
    brinks_dote_scaled = None
    if brinks_show and brinks_table_path is not None:
        brinks_path = Path(brinks_table_path)
        if not brinks_path.exists():
            print(f"[EnergyBudget] Brinks86 table not found at {brinks_path}, skipping grey points.")
        else:
            bdf = pd.read_fwf(brinks_path)
            required_cols = ["R_kpc", "Diam_pc", "Age_Myr", "Mass_1e4Msun", "DV_kms"]
            missing_cols = [c for c in required_cols if c not in bdf.columns]
            if missing_cols:
                print(f"[EnergyBudget] Missing columns in Brinks86 table: {missing_cols}, skipping grey points.")
            else:
                br_r = pd.to_numeric(bdf["R_kpc"], errors="coerce").to_numpy(dtype=float)
                diam_pc = pd.to_numeric(bdf["Diam_pc"], errors="coerce").to_numpy(dtype=float)
                age_myr = pd.to_numeric(bdf["Age_Myr"], errors="coerce").to_numpy(dtype=float)
                mass_1e4 = pd.to_numeric(bdf["Mass_1e4Msun"], errors="coerce").to_numpy(dtype=float)
                dv_kms = np.abs(pd.to_numeric(bdf["DV_kms"], errors="coerce").to_numpy(dtype=float))

                radius_cm = 0.5 * diam_pc * pc2cm
                volume_cm3 = (4.0 / 3.0) * np.pi * radius_cm**3
                age_s = age_myr * SEC_PER_MYR
                mass_g = mass_1e4 * 1e4 * 1.989e33
                v_cms = dv_kms * 1e5
                energy_erg = 0.5 * mass_g * v_cms**2 * BRINKS_ENERGY_SCALE

                with np.errstate(divide="ignore", invalid="ignore"):
                    dotE_phys = energy_erg / (volume_cm3 * age_s)
                dotE_scaled = dotE_phys * 1e28

                mask = np.isfinite(br_r) & np.isfinite(dotE_scaled)
                brinks_age_myr = br_r
                brinks_dote_scaled = dotE_scaled
                if brinks_export_table:
                    export_path = Path(brinks_export_path) if brinks_export_path else brinks_path.parent / "brinks86_dotE_debug.csv"
                    export_df = pd.DataFrame({
                        "Seq": bdf.get("Seq", pd.Series([pd.NA] * len(bdf))),
                        "R_kpc": br_r,
                        "Diam_pc": diam_pc,
                        "Age_Myr": age_myr,
                        "Mass_1e4Msun": mass_1e4,
                        "DV_kms": dv_kms,
                        "dotE_erg_cm3_s": dotE_phys,
                        "dotE_1e-28": dotE_scaled,
                    })
                    export_df.to_csv(export_path, index=False)
                    print(f"[EnergyBudget] Exported Brinks86 derived table to {export_path}")
                if np.any(mask):
                    base_color = brinks_marker_color if brinks_face_color is None else brinks_face_color
                    face_rgba = "none" if float(brinks_face_alpha) == 0 else mcolors.to_rgba(base_color, alpha=brinks_face_alpha)
                    edge_base = brinks_edge_color if brinks_edge_color is not None else brinks_marker_color
                    edge_rgba = mcolors.to_rgba(edge_base, alpha=brinks_edge_alpha)
                    brinks_handle = ax.scatter(
                        br_r[mask],
                        dotE_scaled[mask],
                        s=brinks_marker_size,
                        marker=brinks_marker,
                        facecolors=edge_rgba,
                        edgecolors=edge_rgba,
                        linewidths=max(brinks_marker_lw, 0.0),
                        zorder=3,
                        label="Brinks+86",
                    )
                else:
                    print("[EnergyBudget] No finite Brinks86 R or dotE after parsing; skipping grey points.")

    if brinks_inset:
        _add_brinks_inset(
            ax=ax,
            xvals=brinks_age_myr,
            yvals=brinks_dote_scaled,
            marker_size=brinks_marker_size,
            marker_style=brinks_marker,
            marker_color=brinks_marker_color,
            face_color=brinks_face_color,
            face_alpha=brinks_face_alpha,
            edge_color=brinks_edge_color,
            edge_alpha=brinks_edge_alpha,
            marker_lw=max(brinks_marker_lw, 0.8),
            inset_width=brinks_inset_width,
            inset_height=brinks_inset_height,
            inset_loc=brinks_inset_loc,
            inset_borderpad=brinks_inset_borderpad,
            inset_text=brinks_inset_text,
            inset_text_pos=brinks_inset_text_pos,
            inset_text_fontsize=brinks_inset_text_fontsize,
            tick_length=brinks_inset_tick_length,
            brinks_inset_xlim=brinks_inset_xlim,
            brinks_inset_ylim=brinks_inset_ylim,
        )

    # Legend: single bubble sample + turbulence band
    if show_legend:
        legend_items = []
        legend_labels = []

        # A neutral bubble sample (legend only; actual bubble colors follow colorbar or mono style)
        if color_by_value and cmap is not None and norm is not None:
            sample_fc = cmap(norm(vmin if "vmin" in locals() else 0.5))
            sample_ec = sample_fc
        else:
            base_fc_input = bubble_base_face_color if bubble_base_face_color is not None else (0.6, 0.6, 0.6, 0.6)
            sample_fc = mcolors.to_rgba(base_fc_input, alpha=bubble_marker_face_alpha)
            sample_ec = mcolors.to_rgba(bubble_marker_edge_color, alpha=max(bubble_marker_edge_alpha, 0.1))
        sample_marker = Line2D(
            [], [], marker=bubble_marker, linestyle='None',
            markersize=np.sqrt(bubble_marker_size),
            markerfacecolor=sample_fc,
            markeredgecolor=sample_ec,
            markeredgewidth=bubble_marker_edge_lw,
        )
        legend_items.append(sample_marker)
        legend_labels.append(LEGEND_LABEL_BUBBLES)

        if turb_band is not None:
            legend_items.append(turb_band)
            legend_labels.append(LEGEND_LABEL_TURB)
        if mri_handle is not None:
            legend_items.append(mri_handle)
            legend_labels.append(mri_label)
        if brinks_handle is not None:
            legend_items.append(brinks_handle)
            legend_labels.append(LEGEND_LABEL_BRINKS)
        # order legend entries if requested
        if legend_order:
            order_map = {
                "bubbles": LEGEND_LABEL_BUBBLES,
                "turbulence": LEGEND_LABEL_TURB,
                "brinks": LEGEND_LABEL_BRINKS,
                "mri": mri_label,
            }
            ordered = []
            for key in legend_order:
                lbl = order_map.get(key)
                if lbl in legend_labels:
                    idx = legend_labels.index(lbl)
                    ordered.append((legend_items[idx], legend_labels[idx]))
            # append any remaining not captured
            for itm, lbl in zip(legend_items, legend_labels):
                if lbl not in [l for _, l in ordered]:
                    ordered.append((itm, lbl))
            legend_items = [i for i, _ in ordered]
            legend_labels = [l for _, l in ordered]

        leg = ax.legend(
            legend_items, legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, bbox_to_anchor=(0.05, 0.05, 1, 1)
        )
        if legend_frame:
            leg.get_frame().set_edgecolor(legend_edgecolor)
            leg.get_frame().set_alpha(legend_framealpha)
            leg.get_frame().set_linewidth(legend_edge_lw)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


# -----------------------
# Example call
# -----------------------
if __name__ == "__main__":
    plot_energy_budget_no_mri(
        # axes
        xlim_full=(4, 28),
        ylim=(3e-3, 3e3),

        # turbulence band
        turb_xlim=(3, 33),
        prof_max_path="../data/profile_resampled_max.tsv",
        prof_min_path="../data/profile_resampled_min.tsv",
        turb_color=(0.6, 0.6, 0.6, 0.45),
        turb_edgecolor=(0.1, 0.1, 0.1, 0.85),
        turb_linewidth=0.0,
        turb_zorder=-1000,
        turb_join="intersect",
        turb_extrapolate_union=True,

        # MRI from resampled_med
        mri_show=True,
        mri_profile_path="../data/profile_resampled_med.tsv",
        mri_xlim=(3, 33),
        mri_mode="shade",
        mri_shade_color=(0.7, 0.9, 0.7, 1),
        mri_line_color=(0.4, 0.4, 0.4, 0.95),
        mri_linewidth=0.6,
        mri_linestyle="--",
        mri_label=r"$\mathrm{MRI}$",

        # bubbles and ring table
        bubble_table_path="../code/1113.tab",
        alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
        r_fits="../data/r-m31-jcomb_modHeader.fits",
        ring_table_path="../code/1113.tab",
        nring_column="n_HI_ring_cm-3",

        # continuous coloring by n_ring (auto vmin/vmax)
        color_by_value=False,
        color_cmap_name="coolwarm",
        color_vmin=None,
        color_vmax=None,
        colorbar_label=r"$n_{\rm HI}$[cm$^{-3}$]",

        # circles with data-colored error bars
        bubble_mode="circle",
        bubble_marker_size=BUBBLE_MARKER_SIZE,
        bubble_marker=BUBBLE_MARKER,
        bubble_base_face_color=BUBBLE_BASE_FACE_COLOR,
        bubble_marker_face_alpha=BUBBLE_FACE_ALPHA,
        bubble_use_data_edgecolor=False,
        bubble_marker_edge_color=BUBBLE_EDGE_COLOR,
        bubble_marker_edge_alpha=BUBBLE_EDGE_ALPHA,
        bubble_marker_edge_lw=BUBBLE_EDGE_LW,
        bubble_marker_zorder=6,
        bubble_show_errorbar=BUBBLE_SHOW_ERRORBAR,
        bubble_errorbar_use_data_color=BUBBLE_ERRORBAR_USE_DATA_COLOR,
        bubble_errorbar_data_alpha=BUBBLE_ERRORBAR_DATA_ALPHA,
        bubble_errorbar_elinewidth=BUBBLE_ERRORBAR_ELINEWIDTH,
        bubble_errorbar_capsize=BUBBLE_ERRORBAR_CAPSIZE,
        bubble_errorbar_linestyle=BUBBLE_ERRORBAR_LINESTYLE,
        bubble_errorbar_zorder=BUBBLE_ERRORBAR_ZORDER,

        # legend
        show_legend=True,
        legend_loc="lower left",
        legend_fontsize=8,
        legend_frame=True,
        legend_framealpha=0.9,
        legend_edgecolor="black",

        # output
        output_pdf="dot_e_balance.pdf",
    )
