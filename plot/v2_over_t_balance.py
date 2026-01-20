#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy budget analog using velocity-squared over dissipation time.

This plot mirrors dot_e_balance.py but replaces $\dot{e}$ with
$v_{\rm exp}^2/t_{\rm exp}$ (bubbles) and $\sigma^2/\tau_{\rm diss}$ (turbulence),
both in $(\mathrm{km\,s^{-1}})^2$ Myr$^{-1}$. MRI is omitted because this metric
is not defined for the MRI curve.

Inputs
------
- Bubble table: ../code/1113.tab (fixed-width; uses expansion_vel, radius_pc, ra_pix, dec_pix).
- Ring table: ../code/1113.tab (for n_ring coloring; optional).
- Alpha / R FITS: ../data/alpha-m31-jcomb_modHeader.fits, ../data/r-m31-jcomb_modHeader.fits.
- Turbulence profiles: ../data/profile_resampled_max.tsv and ../data/profile_resampled_min.tsv
  with columns r, x1_mom0, x1_mom2.

Output
------
- v2_over_t_balance.pdf (or the name you pass in).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# -----------------------
# Constants and labels
# -----------------------
PC2CM = 3e18
SEC_PER_MYR = 1e6 * 3.154e7
COSI = np.cos(np.deg2rad(77.0))

TURB_MOM0_COL = "x1_mom0"
TURB_MOM2_COL = "x1_mom2"

CM_METRIC_TO_KM_MYR = SEC_PER_MYR * 1e-10  # convert cm^2 s^-3 to (km/s)^2 Myr^-1

FIG_DPI = 100
SAVEFIG_DPI = 100
FIGSIZE_MAIN = (4.8, 4.8)
AX_LABELSIZE = 10
AX_TITLESIZE = 10
TICK_LABELSIZE = 10
TICK_LENGTH_MAIN = 5
PANEL_LABEL = "b"
PANEL_LABEL_POS = (0.02, 0.98)
PANEL_LABEL_FONTSIZE = 11
PANEL_LABEL_WEIGHT = "bold"
COLORBAR_PAD = 0.01
COLORBAR_ASPECT = 40
COLORBAR_LABELSIZE = 9
COLORBAR_TICKSIZE = 8

BUBBLE_LEGEND_LOC = "lower left"
BUBBLE_LEGEND_FONTSIZE = 7
BUBBLE_LEGEND_FRAMEALPHA = 0.9
BUBBLE_LEGEND_EDGE_COLOR = "black"
BUBBLE_LEGEND_EDGE_LW = 0.5
BUBBLE_LEGEND_ORDER = ("bubbles", "brinks", "turbulence")

BUBBLE_SHOW_BRINKS_DEFAULT = True
BRINKS_TABLE_PATH_DEFAULT = "../data/brinks+86/brinks86_combined.fwf"
BRINKS_MARKER = "^"
BRINKS_MARKER_SIZE = 55
BRINKS_FACE_COLOR = None
BRINKS_FACE_ALPHA = 0.0
BRINKS_EDGE_COLOR = (0.6, 0.6, 0.9)
BRINKS_EDGE_ALPHA = 0.6
BRINKS_EDGE_LW = 0.0
BRINKS_MARKER_COLOR = (0.6, 0.6, 0.9)
BRINKS_LABEL = "Brinks+86"

BUBBLE_MARKER_SIZE = 65
BUBBLE_MARKER = "o"
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)
BUBBLE_FACE_ALPHA = 0.7
BUBBLE_EDGE_ALPHA = 0.0
BUBBLE_EDGE_LW = 0.0
BUBBLE_EDGE_COLOR = (0.9, 0.2, 0.2)
BUBBLE_SHOW_ERRORBAR = True
BUBBLE_ERRORBAR_USE_DATA_COLOR = True
BUBBLE_ERRORBAR_DATA_ALPHA = 0.4
BUBBLE_ERRORBAR_ELINEWIDTH = 0.8
BUBBLE_ERRORBAR_CAPSIZE = 0.0
BUBBLE_ERRORBAR_LINESTYLE = "-"
BUBBLE_ERRORBAR_ZORDER = 5

X_LABEL_MAIN = "Distance to M31's center: $R$ [kpc]"
Y_LABEL_MAIN = r"$v_{\rm exp}^2/t_{\rm exp}$ or $\sigma^2/\tau_{\rm diss}$ [$(\mathrm{km\,s^{-1}})^2$ Myr$^{-1}$]"
LEGEND_LABEL_BUBBLES = "New Bubbles"
LEGEND_LABEL_TURB = "Turbulence" #r"$\sigma^2/\tau_{\rm diss}$"


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


# -----------------------
# Turbulence: sigma^2 / t_diss
# -----------------------
def compute_turb_metric(df: pd.DataFrame, branch: str):
    """
    Compute per-radius turbulent sigma^2 / t_diss.

    df columns: ['r', TURB_MOM0_COL, TURB_MOM2_COL], with 'r' in pc.
    branch: 'max' -> h_pc = 182 - 37 + 13 * r_kpc
            'min' -> h_pc = 182 + 37 + 19 * r_kpc
    return: r_kpc, metric (km^2 s^-2 Myr^-1)
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

    col_den = mom0 * 1.222e6 / 1.42**2 / 3600.0 * 1.823e18
    n_HI = col_den / (2.0 * h_pc * PC2CM / COSI)

    v2 = np.clip(mom2**2 - 0.6 * 8.0**2, 0.0, None)
    sigma = np.sqrt(v2) * 1e5

    with np.errstate(divide="ignore", invalid="ignore"):
        metric_cgs = sigma**3 / (2.0 * h_pc * PC2CM)  # cm^2 s^-3
    metric = metric_cgs * CM_METRIC_TO_KM_MYR
    return r_kpc, metric


def draw_turb_band_metric(
    ax,
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_xlim=(5.0, 24.0),
    color=(0.6, 0.6, 0.6, 0.35),
    edgecolor=(0.1, 0.1, 0.1, 0.85),
    linewidth=0.4,
    zorder=-1000,
    label=r"$\sigma^2/\tau_{\rm diss}$",
    turb_join="intersect",
    extrapolate_union=True,
):
    df_max = pd.read_csv(prof_max_path, sep=r"\t", engine="python")
    df_min = pd.read_csv(prof_min_path, sep=r"\t", engine="python")

    r_kpc_max, rate_max = compute_turb_metric(df_max, branch="max")
    r_kpc_min, rate_min = compute_turb_metric(df_min, branch="min")

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


# -----------------------
# Main plotting function
# -----------------------
def plot_v2_over_t_balance(
    # axes limits
    xlim_full=(4.0, 28.0),
    ylim=(1e-2, 2e3),

    # turbulence band input and style
    turb_xlim=(5.0, 24.0),
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_color=(0.6, 0.6, 0.6, 0.35),
    turb_edgecolor=(0.1, 0.1, 0.1, 0.85),
    turb_linewidth=0.4,
    turb_zorder=-1000,
    turb_label=r"$\sigma^2/\tau_{\rm diss}$",
    turb_join="intersect",
    turb_extrapolate_union=True,

    # bubbles input
    bubble_table_path="../code/1113.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",

    # ring table and columns (for color)
    ring_table_path="../code/1113.tab",
    nring_column="n_HI_ring_cm-3",

    # continuous color map settings for n_ring
    color_by_value=False,
    color_cmap_name="coolwarm",
    color_vmin=None,   # if None, auto from data percentiles
    color_vmax=None,   # if None, auto from data percentiles
    colorbar_label=r"$n_{\rm HI}$ in ring [cm$^{-3}$]",
    colorbar_aspect=COLORBAR_ASPECT,

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
    legend_loc=BUBBLE_LEGEND_LOC,
    legend_fontsize=BUBBLE_LEGEND_FONTSIZE,
    legend_frame=True,
    legend_framealpha=BUBBLE_LEGEND_FRAMEALPHA,
    legend_edgecolor=BUBBLE_LEGEND_EDGE_COLOR,
    legend_edge_lw=BUBBLE_LEGEND_EDGE_LW,
    legend_order=BUBBLE_LEGEND_ORDER,

    # Brinks+86 overlay
    brinks_table_path=BRINKS_TABLE_PATH_DEFAULT,
    brinks_show=BUBBLE_SHOW_BRINKS_DEFAULT,
    brinks_marker_size=BRINKS_MARKER_SIZE,
    brinks_marker=BRINKS_MARKER,
    brinks_marker_color=BRINKS_MARKER_COLOR,
    brinks_face_color=BRINKS_FACE_COLOR,
    brinks_face_alpha=BRINKS_FACE_ALPHA,
    brinks_edge_color=BRINKS_EDGE_COLOR,
    brinks_edge_alpha=BRINKS_EDGE_ALPHA,
    brinks_marker_lw=BRINKS_EDGE_LW,
    brinks_label=BRINKS_LABEL,

    # output
    output_pdf="v2_over_t_balance.pdf",
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
    Bubbles are colored continuously by n_ring using a linear Normalize.
    n_ring from ring_table_path also enters the color (not the metric).
    """

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
    ax.text(
        PANEL_LABEL_POS[0],
        PANEL_LABEL_POS[1],
        PANEL_LABEL,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight=PANEL_LABEL_WEIGHT,
    )
    ax.tick_params(direction="in", which="both", length=tick_length_main, top=True, bottom=True, left=True, right=True)
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Turbulence band
    turb_band = draw_turb_band_metric(
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
    df_bubble = pd.read_fwf(bubble_table_path, header=0, infer_nrows=int(1e6))
    ra_pix = df_bubble["ra_pix"]
    dec_pix = df_bubble["dec_pix"]
    r_bub_pc = df_bubble["radius_pc"].values
    vexp_kms = df_bubble["expansion_vel"].values + 1  # small offset, consistent with dot_e_balance

    angles = get_fits_at_pix(alpha_fits, ra_pix, dec_pix) - 90.0
    x_pos = get_fits_at_pix(r_fits, ra_pix, dec_pix) / 1e3  # kpc

    # v^2 / t_exp for bubbles
    vexp_cms = vexp_kms * 1e5
    t_exp = 0.6 * r_bub_pc * PC2CM / vexp_cms
    with np.errstate(divide="ignore", invalid="ignore"):
        metric_bub_cgs = vexp_cms**2 / t_exp
    metric_bub = metric_bub_cgs * CM_METRIC_TO_KM_MYR

    # nring for coloring
    df_ring = pd.read_fwf(ring_table_path, header=0, infer_nrows=int(1e6))
    if nring_column not in df_ring.columns:
        raise KeyError(f"'{nring_column}' not found in {ring_table_path}")
    nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)

    if len(df_ring) != len(df_bubble):
        raise ValueError(
            f"Row count mismatch: ring table ({len(df_ring)}) vs bubble table ({len(df_bubble)}). "
            f"Please ensure the two tables are aligned row by row or add a common 'id' to join."
        )

    # print min and max
    def _print_minmax(name, arr):
        arr = np.asarray(arr, dtype=float)
        m = np.isfinite(arr)
        if not np.any(m):
            print(f"[v2/t] {name}: no finite values")
            return
        print(f"[v2/t] {name}: min={np.nanmin(arr[m]):.6g}, max={np.nanmax(arr[m]):.6g}, N={np.count_nonzero(m)}")

    _print_minmax("R_kpc (x_pos)", x_pos)
    _print_minmax("v2/t (y)", metric_bub)
    _print_minmax("n_ring [cm^-3]", nring)

    # Brinks+86 overlay (computed in the same metric)
    brinks_handle = None
    if brinks_show and brinks_table_path is not None:
        bpath = Path(brinks_table_path)
        if bpath.exists():
            bdf = pd.read_fwf(bpath)
            required_cols = ["R_kpc", "Diam_pc", "DV_kms", "Mass_1e4Msun"]
            missing_cols = [c for c in required_cols if c not in bdf.columns]
            if missing_cols:
                print(f"[v2/t] Missing columns in Brinks86 table: {missing_cols}, skipping Brinks overlay.")
            else:
                br_r = pd.to_numeric(bdf["R_kpc"], errors="coerce").to_numpy(dtype=float)
                diam_pc = pd.to_numeric(bdf["Diam_pc"], errors="coerce").to_numpy(dtype=float)
                dv_kms = np.abs(pd.to_numeric(bdf["DV_kms"], errors="coerce").to_numpy(dtype=float))
                radius_pc = 0.5 * diam_pc
                vexp_cms_b = dv_kms * 1e5
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_b = 0.6 * radius_pc * PC2CM / vexp_cms_b
                    metric_b_cgs = vexp_cms_b**2 / t_b
                metric_b = metric_b_cgs * CM_METRIC_TO_KM_MYR

                mask_b = np.isfinite(br_r) & np.isfinite(metric_b)
                if np.any(mask_b):
                    edge_base = brinks_edge_color if brinks_edge_color is not None else brinks_marker_color
                    edge_rgba = mcolors.to_rgba(edge_base, alpha=brinks_edge_alpha)
                    brinks_handle = ax.scatter(
                        br_r[mask_b],
                        metric_b[mask_b],
                        s=brinks_marker_size,
                        marker=brinks_marker,
                        facecolor=edge_rgba,
                        edgecolor=edge_rgba,
                        linewidths=max(brinks_marker_lw, 0.0),
                        zorder=12,
                        label=brinks_label,
                    )
                else:
                    print("[v2/t] No finite Brinks R or metric after parsing; skipping Brinks overlay.")
        else:
            print(f"[v2/t] Brinks86 table not found at {bpath}, skipping Brinks overlay.")

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
            rect = plt.Rectangle(
                (x_pos[i] - dxk, metric_bub[i] * 0.5),
                2 * dxk, metric_bub[i],
                facecolor=fc,
                edgecolor=ec,
                lw=bubble_rect_edge_lw,
                zorder=bubble_rect_zorder
            )
            ax.add_patch(rect)

    elif bubble_mode == "circle":
        data_colors = nring_colors
        facecolors = np.stack([np.r_[c[:3], bubble_marker_face_alpha] for c in data_colors])

        ax.scatter(
            x_pos, metric_bub,
            s=bubble_marker_size,
            marker=bubble_marker,
            facecolor=facecolors,
            edgecolor="none",
            linewidths=0,
            zorder=bubble_marker_zorder
        )

        # error bars
        if bubble_show_errorbar:
            h_i = 182 + 16 * x_pos
            dh = h_i * np.tan(np.deg2rad(77)) * np.cos(np.deg2rad(angles))
            dr = r_bub_pc
            dxk = np.sqrt(dh**2 + dr**2) / 1e3
            xerr = dxk
            yerr = 0.5 * metric_bub

            if bubble_errorbar_use_data_color:
                for xi, yi, xe, ye, col in zip(x_pos, metric_bub, xerr, yerr, data_colors):
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
                    x_pos, metric_bub,
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

    # set axes limits
    ax.set_xlim(xlim_full)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)

    # Legend: single bubble sample + turbulence band
    if show_legend:
        legend_entries = {}

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
        legend_entries["bubbles"] = (sample_marker, LEGEND_LABEL_BUBBLES)

        if turb_band is not None:
            legend_entries["turbulence"] = (turb_band, LEGEND_LABEL_TURB)
        if brinks_handle is not None:
            legend_entries["brinks"] = (brinks_handle, brinks_label)

        if legend_order:
            ordered = [(legend_entries[k]) for k in legend_order if k in legend_entries]
        else:
            ordered = list(legend_entries.values())

        legend_items = [i for i, _ in ordered]
        legend_labels = [l for _, l in ordered]

        leg = ax.legend(
            legend_items, legend_labels,
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
    plot_v2_over_t_balance(
        # axes
        xlim_full=(4, 28),
        ylim=(1e-1, 2e3),

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

        # bubbles and ring table
        bubble_table_path="../code/1113.tab",
        alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
        r_fits="../data/r-m31-jcomb_modHeader.fits",
        ring_table_path="../code/1113.tab",
        nring_column="n_HI_ring_cm-3",

        # Brinks+86 overlay
        brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
        brinks_show=True,
        brinks_marker_size=BRINKS_MARKER_SIZE,
        brinks_marker=BRINKS_MARKER,
        brinks_marker_color=BRINKS_MARKER_COLOR,
        brinks_face_color=BRINKS_FACE_COLOR,
        brinks_face_alpha=BRINKS_FACE_ALPHA,
        brinks_edge_color=BRINKS_EDGE_COLOR,
        brinks_edge_alpha=BRINKS_EDGE_ALPHA,
        brinks_marker_lw=BRINKS_EDGE_LW,
        brinks_label=BRINKS_LABEL,

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
        legend_loc=BUBBLE_LEGEND_LOC,
        legend_fontsize=BUBBLE_LEGEND_FONTSIZE,
        legend_frame=True,
        legend_framealpha=BUBBLE_LEGEND_FRAMEALPHA,
        legend_edgecolor=BUBBLE_LEGEND_EDGE_COLOR,

        # output
        output_pdf="v2_over_t_balance.pdf",
    )
