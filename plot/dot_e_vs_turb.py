#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dot-e bubble vs turbulence scatter (bubble dotE vs local turbulent dotE).

Inputs
------
- bubble_table_path : Fixed-width bubble table (default: ../code/1113.tab).
- ring_table_path   : Ring table providing n_HI_ring_cm-3 (default: ../code/1113.tab).
- alpha_fits        : Angle map FITS for geometry (default: ../data/alpha-m31-jcomb_modHeader.fits).
- r_fits            : Deprojected radius map FITS in pc (default: ../data/r-m31-jcomb_modHeader.fits).
- prof_max_path     : Resampled profile TSV (max branch) with columns ['r', x1_mom0, x1_mom2].
- prof_min_path     : Resampled profile TSV (min branch) with columns ['r', x1_mom0, x1_mom2].

Outputs
-------
- dot_e_vs_turb.pdf : Scatter of bubble dotE (y) vs turbulent dotE at same R (x).
- Optional CSV (if export_table=True) with per-bubble values.

Notes
-----
- dotE_bubble follows dot_e_balance: dotE = 0.5 * rho * v^3 / (h * pc2cm) in 10^-28 units,
  with h_pc = 182 + 16 * R_kpc, rho = 1.673e-24 * n_ring.
- Turbulent dotE is interpolated at each bubble radius from the max/min profile curves.
  You can choose the combination via turb_value_mode: 'mean', 'min', or 'max'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from pathlib import Path

# constants
pc2cm = 3e18
COSI = np.cos(np.deg2rad(77.0))

TURB_MOM0_COL = "x1_mom0"
TURB_MOM2_COL = "x1_mom2"
BUBBLE_MARKER_SIZE = 65
BUBBLE_MARKER = "o"
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)
BUBBLE_FACE_ALPHA = 0.7
BUBBLE_EDGE_ALPHA = 0.0
BUBBLE_EDGE_LW = 0.0
BUBBLE_EDGE_COLOR = (0.9, 0.2, 0.2)
BUBBLE_SHOW_ERRORBAR = False
BUBBLE_ERRORBAR_USE_DATA_COLOR = True
BUBBLE_ERRORBAR_DATA_ALPHA = 0.4
BUBBLE_ERRORBAR_ELINEWIDTH = 0.6
BUBBLE_ERRORBAR_CAPSIZE = 0.0
BUBBLE_ERRORBAR_LINESTYLE = "-"
BUBBLE_ERRORBAR_ZORDER = 5

BRINKS_MARKER = "^"
BRINKS_MARKER_SIZE = 55
BRINKS_FACE_COLOR = None
BRINKS_FACE_ALPHA = 0.6
BRINKS_EDGE_COLOR = (0.6, 0.6, 0.9)
BRINKS_EDGE_ALPHA = 0.6
BRINKS_EDGE_LW = 0.0
BRINKS_MARKER_COLOR = (0.6, 0.6, 0.9)
BRINKS_SHOW = True
BRINKS_ENERGY_SCALE = 0.44
BRINKS_INSET_ENABLED = True
BRINKS_INSET_LOC = "lower right"
BRINKS_INSET_SIZE = "30%"
BRINKS_INSET_BORDERPAD = 0.8
BRINKS_INSET_TEXT = "Brinks+86 (full)"
BRINKS_INSET_TEXT_POS = (0.25, 0.95)
BRINKS_INSET_TEXT_FONTSIZE = 7
BRINKS_INSET_TICK_LENGTH = 3
BRINKS_INSET_XLIM = (1e-1, 1e2)
BRINKS_INSET_YLIM = (1e-3, 1e7)
BRINKS_INSET_BBOX = (-0.0, 0.05, 1.0, 1.0)
SIGMA_LOG_SHOW = True
SIGMA_LOG_LABEL = r"$\Delta_{1:1}$"
FIGSIZE_MAIN = (4.8, 4.8)
LEGEND_BBOX = (0.05, -0.02, 1.0, 1.0)
LEGEND_LOC = "upper left"
LEGEND_FONTSIZE = 8
LEGEND_FRAME = True
LEGEND_FRAMEALPHA = 0.9
LEGEND_EDGE_COLOR = "black"
LEGEND_EDGE_LW = 0.5
LEGEND_ORDER = ("bubbles", "brinks", "turbulence", "one_to_one", "sigma") # e.g., ("bubbles", "turbulence", "brinks", "one_to_one", "sigma")
X_LABEL_MAIN = r"$\dot{e}_{\rm turb}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]"
Y_LABEL_MAIN = r"$\dot{e}_{\rm bubble}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]"
LEGEND_LABEL_ONE_TO_ONE = "1:1"
LEGEND_LABEL_BRINKS = "Brinks+86"
LEGEND_LABEL_BUBBLES = "New Bubbles"


def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


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


def interpolate_turb_at_r(r_grid, rate_grid, r_targets):
    """Interpolate turbulent rate at target radii."""
    return np.interp(r_targets, r_grid, rate_grid, left=np.nan, right=np.nan)

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


def compute_bubble_dote(
    bubble_table_path,
    ring_table_path,
    alpha_fits,
    r_fits,
    nring_column="n_HI_ring_cm-3",
):
    """Compute bubble dotE and positions using the same prescription as dot_e_balance."""
    df_bubble = pd.read_fwf(bubble_table_path, header=0, infer_nrows=int(1e6))
    df_ring = pd.read_fwf(ring_table_path, header=0, infer_nrows=int(1e6))
    if nring_column not in df_ring.columns:
        raise KeyError(f"'{nring_column}' not found in {ring_table_path}")
    if len(df_ring) != len(df_bubble):
        raise ValueError(
            f"Row count mismatch: ring table ({len(df_ring)}) vs bubble table ({len(df_bubble)}). "
            "Ensure alignment or add a join key."
        )

    ra_pix = df_bubble["ra_pix"]
    dec_pix = df_bubble["dec_pix"]
    r_bub_pc = df_bubble["radius_pc"].values
    vexp_kms = df_bubble["expansion_vel"].values + 1  # small offset consistent with dot_e_balance

    angles = get_fits_at_pix(alpha_fits, ra_pix, dec_pix) - 90.0
    x_pos_pc = get_fits_at_pix(r_fits, ra_pix, dec_pix)
    x_pos_kpc = x_pos_pc / 1e3

    nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)

    h_pc = 182 + 16 * x_pos_kpc
    rho = 1.673e-24 * nring
    v_cms = (vexp_kms * 1e5)
    dotE = 0.5 * rho * v_cms**3 / (h_pc * pc2cm) * 1e28  # in 10^-28 units

    return {
        "R_kpc": x_pos_kpc,
        "dotE_bubble": dotE,
        "nring": nring,
        "angles": angles,
        "r_bub_pc": r_bub_pc,
    }


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
    iax.yaxis.set_minor_locator(mticker.NullLocator())
    iax.tick_params(direction="in", which="both", length=inset_tick_length, labelsize=inset_text_fontsize, top=True, right=True)
    iax.xaxis.set_minor_locator(mticker.NullLocator())
    iax.yaxis.set_minor_locator(mticker.NullLocator())
    iax.text(
        inset_text_pos[0],
        inset_text_pos[1],
        inset_text,
        ha="left",
        va="top",
        fontsize=inset_text_fontsize,
        transform=iax.transAxes,
    )


def plot_dot_e_vs_turb(
    # axes
    xlim=(1e-2, 1e3),
    ylim=(1e-2, 1e3),

    # inputs
    bubble_table_path="../code/1113.tab",
    ring_table_path="../code/1113.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",
    nring_column="n_HI_ring_cm-3",
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_value_mode="mean",  # 'mean', 'min', 'max'

    # styles
    color_by_radius=False,
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
    cmap_name="coolwarm",
    color_vmin=None,
    color_vmax=None,
    colorbar_label=r"$R$ [kpc]",

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
    sigma_log_show=SIGMA_LOG_SHOW,
    sigma_log_label=SIGMA_LOG_LABEL,
    legend_loc=LEGEND_LOC,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_frame=LEGEND_FRAME,
    legend_framealpha=LEGEND_FRAMEALPHA,
    legend_edgecolor=LEGEND_EDGE_COLOR,
    legend_edge_lw=LEGEND_EDGE_LW,
    legend_order=LEGEND_ORDER,
    figsize=FIGSIZE_MAIN,

    # export
    export_table=False,
    export_path="dot_e_vs_turb.csv",

    # output
    output_pdf="dot_e_vs_turb.pdf",
):
    """Scatter of bubble dotE vs turbulent dotE at bubble radii."""

    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL_MAIN)
    ax.set_ylabel(Y_LABEL_MAIN)
    ax.tick_params(direction="in", which="both", length=5, top=True, right=True)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.text(0.02, 0.98, "c", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    # load turbulence profiles
    df_max = pd.read_csv(prof_max_path, sep=r"\t", engine="python")
    df_min = pd.read_csv(prof_min_path, sep=r"\t", engine="python")
    r_kpc_max, rate_max = compute_turb_from_profile_df(df_max, branch="max")
    r_kpc_min, rate_min = compute_turb_from_profile_df(df_min, branch="min")

    # bubble dotE
    bubble = compute_bubble_dote(
        bubble_table_path=bubble_table_path,
        ring_table_path=ring_table_path,
        alpha_fits=alpha_fits,
        r_fits=r_fits,
        nring_column=nring_column,
    )

    # interpolate turbulence at bubble R
    turb_at_r_max = interpolate_turb_at_r(r_kpc_max, rate_max, bubble["R_kpc"])
    turb_at_r_min = interpolate_turb_at_r(r_kpc_min, rate_min, bubble["R_kpc"])
    if turb_value_mode == "mean":
        turb_at_r = 0.5 * (turb_at_r_max + turb_at_r_min)
    elif turb_value_mode == "max":
        turb_at_r = turb_at_r_max
    elif turb_value_mode == "min":
        turb_at_r = turb_at_r_min
    else:
        raise ValueError("turb_value_mode must be 'mean', 'min', or 'max'.")

    # color mapping by radius (optional)
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
    edgecolors = None

    # filter finite points
    mask = np.isfinite(turb_at_r) & np.isfinite(bubble["dotE_bubble"])
    ax.scatter(
        turb_at_r[mask],
        bubble["dotE_bubble"][mask],
        s=bubble_marker_size,
        marker=bubble_marker,
        facecolor=facecolors[mask],
        edgecolor="none",
        linewidths=0,
        zorder=bubble_marker_zorder,
    )

    if bubble_show_errorbar:
        xerr = 0.05 * turb_at_r
        yerr = 0.5 * bubble["dotE_bubble"]
        if bubble_errorbar_use_data_color and color_by_radius and cmap is not None and norm is not None:
            for xi, yi, xe, ye, col in zip(turb_at_r[mask], bubble["dotE_bubble"][mask], xerr[mask], yerr[mask], colors[mask]):
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
                turb_at_r,
                bubble["dotE_bubble"],
                xerr=xerr,
                yerr=yerr,
                fmt='none',
                ecolor=bubble_errorbar_color,
                elinewidth=bubble_errorbar_elinewidth,
                capsize=bubble_errorbar_capsize,
                ls=bubble_errorbar_linestyle,
                zorder=bubble_errorbar_zorder
            )

    # 1:1 line
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

    sigma_log = _sigma_log(turb_at_r[mask], bubble["dotE_bubble"][mask])
    if sigma_log_show and np.isfinite(sigma_log):
        leg_sigma = Line2D([], [], marker="None", linestyle="None", label=f"{sigma_log_label} = {sigma_log:.2f} dex")
    else:
        leg_sigma = None

    # Brinks+86 overlay as crosses
    brinks_handle = None
    brinks_x = None
    brinks_y = None
    if brinks_show and brinks_table_path is not None:
        bpath = Path(brinks_table_path)
        if bpath.exists():
            bdf = pd.read_fwf(bpath)
            required = ["R_kpc", "Diam_pc", "Age_Myr", "Mass_1e4Msun", "DV_kms"]
            missing = [c for c in required if c not in bdf.columns]
            if not missing:
                r_b = pd.to_numeric(bdf["R_kpc"], errors="coerce").to_numpy(dtype=float)
                diam_pc = pd.to_numeric(bdf["Diam_pc"], errors="coerce").to_numpy(dtype=float)
                age_myr = pd.to_numeric(bdf["Age_Myr"], errors="coerce").to_numpy(dtype=float)
                mass_1e4 = pd.to_numeric(bdf["Mass_1e4Msun"], errors="coerce").to_numpy(dtype=float)
                dv_kms = np.abs(pd.to_numeric(bdf["DV_kms"], errors="coerce").to_numpy(dtype=float))

                radius_cm = 0.5 * diam_pc * pc2cm
                volume_cm3 = (4.0 / 3.0) * np.pi * radius_cm**3
                age_s = age_myr * 1e6 * 3.154e7
                mass_g = mass_1e4 * 1e4 * 1.989e33
                v_cms = dv_kms * 1e5
                energy_erg = 0.5 * mass_g * v_cms**2 * BRINKS_ENERGY_SCALE

                with np.errstate(divide="ignore", invalid="ignore"):
                    dote_b = energy_erg / (volume_cm3 * age_s)
                dote_b_scaled = dote_b * 1e28

                turb_b = interpolate_turb_at_r(r_kpc_max, rate_max, r_b)
                turb_b2 = interpolate_turb_at_r(r_kpc_min, rate_min, r_b)
                if turb_value_mode == "mean":
                    turb_both = 0.5 * (turb_b + turb_b2)
                elif turb_value_mode == "max":
                    turb_both = turb_b
                else:
                    turb_both = turb_b2

                mb = np.isfinite(turb_both) & np.isfinite(dote_b_scaled)
                brinks_x = turb_both
                brinks_y = dote_b_scaled
                if np.any(mb):
                    edge_base = brinks_edge_color if brinks_edge_color is not None else brinks_marker_color
                    edge_rgba = mcolors.to_rgba(edge_base, alpha=brinks_edge_alpha)
                    brinks_handle = ax.scatter(
                        turb_both[mb],
                        dote_b_scaled[mb],
                        marker=brinks_marker,
                        s=brinks_marker_size,
                        facecolor=edge_rgba,
                        edgecolor=edge_rgba,
                        linewidths=max(brinks_marker_lw, 0.0),
                        zorder=12,
                        label=LEGEND_LABEL_BRINKS,
                    )
        else:
            print(f"[DotEvsTurb] Brinks table not found at {bpath}, skipping crosses.")

    # colorbar
    if color_by_radius and cmap is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)
        cbar.set_label(colorbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)

    if brinks_inset and brinks_x is not None and brinks_y is not None:
        # widen inset limits relative to main axes if not provided
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

    handles, labels = ax.get_legend_handles_labels()
    if leg_sigma is not None:
        handles.append(leg_sigma)
        labels.append(f"{sigma_log_label} = {sigma_log:.2f} dex")

    # Ensure bubble sample appears with fixed label
    bubble_sample = Line2D(
        [], [], marker=bubble_marker, linestyle='None',
        markersize=np.sqrt(bubble_marker_size),
        markerfacecolor=mcolors.to_rgba(bubble_marker_edge_color if bubble_base_face_color is None else bubble_base_face_color, alpha=bubble_marker_face_alpha),
        markeredgecolor=mcolors.to_rgba(bubble_marker_edge_color, alpha=max(bubble_marker_edge_alpha, 0.1)),
        markeredgewidth=bubble_marker_edge_lw,
        label=LEGEND_LABEL_BUBBLES,
    )
    handles.append(bubble_sample)
    labels.append(LEGEND_LABEL_BUBBLES)

    if legend_order:
        key_map = {
            "bubbles": LEGEND_LABEL_BUBBLES,
            "one_to_one": LEGEND_LABEL_ONE_TO_ONE,
            "brinks": LEGEND_LABEL_BRINKS,
            "sigma": SIGMA_LOG_LABEL,
        }
        order_map = {lbl: hdl for hdl, lbl in zip(handles, labels)}
        ordered = []
        for key in legend_order:
            lbl = key_map.get(key, key)
            if lbl in order_map:
                ordered.append((order_map[lbl], lbl))
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
            "dotE_turb_1e-28": turb_at_r,
            "dotE_bubble_1e-28": bubble["dotE_bubble"],
            "nring_cm-3": bubble["nring"],
        })
        export_path = Path(export_path)
        export_df.to_csv(export_path, index=False)
        print(f"[DotEvsTurb] Exported table to {export_path}")

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    plot_dot_e_vs_turb()
