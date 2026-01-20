#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Five-panel bubble properties vs R (shared x), aligned like your FITS multi-panel layout.

Panels (top→bottom):
  (a) r [pc] vs R [kpc]
  (b) v_exp [km/s] vs R [kpc]
  (c) age = 0.5 * r / v [Myr] vs R [kpc]
  (d) n_HI [cm^-3] vs R [kpc]  (the quantity previously used for coloring)
  (e) N_SN (from column 'SN_weaver') vs R [kpc] on a log y axis

Marker style:
- Fixed red markers, edge-free, no data-driven coloring and no colorbar.

Layout:
- Use fig.add_axes with fixed boxes for perfect alignment.
- Only bottom panel (e) shows x tick labels; y labels are shown on all panels.
- Panel tags "(a) (b) (c) (d) (e)" at top-left corners.

Extra for panel (e):
- Y axis is log scale.
- If sn_ylim is None, Y range is chosen automatically from data and y-errors,
  with some padding so points and errorbars do not touch the frame.
"""

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from astropy.io import fits

# -----------------------
# Constants
# -----------------------
pc2cm = 3.08567758128e18
mH    = 1.6735575e-24
MU    = 1.4
COSI  = float(np.cos(np.deg2rad(77.0)))
PC_PER_MYR_PER_KMS = 1.0227121650537077  # 1 km/s = 1.0227 pc/Myr
ESN   = 1.0e51

BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)


# -----------------------
# Utilities
# -----------------------
def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan, dtype=float)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def compute_SN_per_bubble(r_pc, v_kms, NHI, h_pc):
    r_cm = r_pc * pc2cm
    v = v_kms * 1e5
    nHI = NHI / (2*h_pc*pc2cm/COSI)
    rho = MU*mH*nHI
    Eb = (4.0/3.0)*np.pi*(r_cm**3)*0.5*rho*(v**2)
    return Eb/ESN*(11.0/3.0)


def _prep_bubble_data(bubble_table_path, r_fits_path, alpha_fits_path, mom0_fits_path):
    df = pd.read_fwf(bubble_table_path, header=0, infer_nrows=int(1e6))

    bx = df["ra_pix"].to_numpy()
    by = df["dec_pix"].to_numpy()
    r_pc   = df["radius_pc"].to_numpy(float)
    v_kms  = df["expansion_vel"].to_numpy(float) + 1.0  # keep consistent

    R_pc  = get_fits_at_pix(r_fits_path, bx, by)
    R_kpc = R_pc / 1.0e3

    angles = get_fits_at_pix(alpha_fits_path, bx, by) - 90.0
    mom0   = get_fits_at_pix(mom0_fits_path, bx, by)

    # Column density and derived quantities
    NHI = mom0 * 1.222e6/1.42**2/3600.0 * 1.823e18
    h_pc = 182.0 + 16.0 * R_kpc
    nHI_derived  = NHI / (2.0*h_pc*pc2cm/COSI)

    # Use table column n_HI_ring_cm-3 for coloring if present
    if "n_HI_ring_cm-3" in df.columns:
        n_ring = df["n_HI_ring_cm-3"].to_numpy(float)
    else:
        n_ring = nHI_derived

    # Total SN from Weaver-like energy, kept for reference (not used for coloring)
    total_SN = compute_SN_per_bubble(r_pc, v_kms, NHI, h_pc)

    # N_SN from Weaver column in the table
    if "SN_weaver" in df.columns:
        NSN_weaver = df["SN_weaver"].to_numpy(float)
    elif "SN_num" in df.columns:
        NSN_weaver = df["SN_num"].to_numpy(float)
    else:
        NSN_weaver = total_SN

    # x-err like EnergyBudget: dx = sqrt(dh^2 + r^2)/1e3
    dh   = h_pc * np.tan(np.deg2rad(77.0)) * np.cos(np.deg2rad(angles))
    dxk  = np.sqrt(dh**2 + r_pc**2) / 1.0e3  # kpc

    # age (Myr)
    age_myr = 0.5 * r_pc / (v_kms * PC_PER_MYR_PER_KMS)

    return dict(
        R_kpc=R_kpc,
        r_pc=r_pc,
        v_kms=v_kms,
        age_myr=age_myr,
        angles=angles,
        h_pc=h_pc,
        nHI_derived=nHI_derived,
        NHI=NHI,
        total_SN=total_SN,
        xerr_kpc=dxk,
        n_ring=n_ring,
        NSN_weaver=NSN_weaver
    )


# -----------------------
# Bubble renderers (EnergyBudget-style)
# -----------------------
def _draw_bubbles_rect(ax, x_pos, y_val, width_half_kpc, color_vals,
                       cmap, norm,
                       rect_face_alpha, rect_edge_alpha, rect_edge_lw, rect_zorder):
    for xi, yi, dx, cv in zip(x_pos, y_val, width_half_kpc, color_vals):
        base_color = cmap(norm(cv)) if np.isfinite(cv) else (0.3, 0.3, 0.3, 0.2)
        fc = base_color[:3] + (rect_face_alpha,)
        ec = base_color[:3] + (rect_edge_alpha,)
        rect = Rectangle(
            (xi - dx, yi * 0.5),
            2.0 * dx, yi,
            facecolor=fc,
            edgecolor=ec,
            lw=rect_edge_lw,
            zorder=rect_zorder
        )
        ax.add_patch(rect)


def _draw_bubbles_circle(ax, x_pos, y_val, color_vals, xerr, yerr,
                         cmap, norm,
                         marker_size, face_alpha,
                         use_data_edgecolor, marker_edge_color, marker_edge_alpha, marker_edge_lw,
                         show_errorbar, errorbar_use_data_color, errorbar_data_alpha,
                         errorbar_color, errorbar_elinewidth, errorbar_capsize, errorbar_linestyle,
                         marker_zorder, errorbar_zorder):
    # face and edge colors
    data_colors = np.array([cmap(norm(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 0.3)
                            for v in color_vals])
    facecolors  = np.stack([np.r_[c[:3], face_alpha] for c in data_colors])

    if use_data_edgecolor:
        edgecolors = np.stack([np.r_[c[:3], marker_edge_alpha] for c in data_colors])
    else:
        ec = np.r_[marker_edge_color[:3], marker_edge_alpha]
        edgecolors = np.repeat(ec[None, :], len(x_pos), axis=0)

    ax.scatter(
        x_pos, y_val,
        s=marker_size,
        facecolor=facecolors,
        edgecolor=edgecolors,
        linewidths=marker_edge_lw,
        zorder=marker_zorder
    )

    if show_errorbar:
        # Simple symmetric errors for now: xerr and yerr are 1D or None
        if xerr is not None:
            xerr_arr = np.asarray(xerr)
        else:
            xerr_arr = None

        if yerr is not None:
            yerr_arr = np.asarray(yerr)
        else:
            yerr_arr = None

        if errorbar_use_data_color:
            for i, (xi, yi, col) in enumerate(zip(x_pos, y_val, data_colors)):
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    continue

                if xerr_arr is None:
                    xe = None
                else:
                    xe_val = xerr_arr[i]
                    xe = [[xe_val], [xe_val]]

                if yerr_arr is None:
                    ye = None
                else:
                    ye_val = yerr_arr[i]
                    ye = [[ye_val], [ye_val]]

                ecolor = (col[0], col[1], col[2], float(errorbar_data_alpha))
                ax.errorbar(
                    [xi], [yi],
                    xerr=xe,
                    yerr=ye,
                    fmt='none',
                    ecolor=ecolor,
                    elinewidth=errorbar_elinewidth,
                    capsize=errorbar_capsize,
                    ls=errorbar_linestyle,
                    zorder=errorbar_zorder
                )
        else:
            if xerr_arr is None:
                xerr_plot = None
            else:
                xerr_plot = np.vstack([xerr_arr, xerr_arr])
            if yerr_arr is None:
                yerr_plot = None
            else:
                yerr_plot = np.vstack([yerr_arr, yerr_arr])

            ax.errorbar(
                x_pos, y_val,
                xerr=xerr_plot, yerr=yerr_plot,
                fmt='none',
                ecolor=errorbar_color,
                elinewidth=errorbar_elinewidth,
                capsize=errorbar_capsize,
                ls=errorbar_linestyle,
                zorder=errorbar_zorder
            )


def _draw_bubbles_fixed_red(
    ax,
    x_pos,
    y_val,
    *,
    xerr=None,
    yerr=None,
    marker_size=45.0,
    face_alpha=0.70,
    show_errorbar=True,
    errorbar_color=(0.0, 0.0, 0.0, 0.35),
    errorbar_elinewidth=0.6,
    errorbar_capsize=0.0,
    errorbar_linestyle='-',
    marker_zorder=6,
    errorbar_zorder=5,
):
    x = np.asarray(x_pos, dtype=float)
    y = np.asarray(y_val, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return

    x = x[m]
    y = y[m]

    ax.scatter(
        x,
        y,
        s=marker_size,
        facecolor=(BUBBLE_BASE_FACE_COLOR[0], BUBBLE_BASE_FACE_COLOR[1], BUBBLE_BASE_FACE_COLOR[2], face_alpha),
        edgecolor="none",
        linewidths=0.0,
        zorder=marker_zorder,
    )

    if show_errorbar and (xerr is not None or yerr is not None):
        xe = None if xerr is None else np.asarray(xerr, dtype=float)[m]
        ye = None if yerr is None else np.asarray(yerr, dtype=float)[m]
        ax.errorbar(
            x,
            y,
            xerr=xe,
            yerr=ye,
            fmt="none",
            ecolor=errorbar_color,
            elinewidth=errorbar_elinewidth,
            capsize=errorbar_capsize,
            ls=errorbar_linestyle,
            zorder=errorbar_zorder,
        )


# -----------------------
# Main
# -----------------------
def plot_bubble_r_v_age_aligned(
    # inputs
    bubble_table_path="../code/0709-decon_hb.tab",
    r_fits_path="../data/r-m31-jcomb_modHeader.fits",
    alpha_fits_path="../data/alpha-m31-jcomb_modHeader.fits",
    mom0_fits_path="../data/jcomb_vcube_scaled2_mom0.fits",

    # layout parameters
    fig_size=(6.1, 9.4),
    left=0.12, width=0.78,
    panel_height=0.155,
    bottoms=(0.80, 0.62, 0.44, 0.26, 0.08),   # (a),(b),(c),(d),(e) from top to bottom
    colorbar_pad=0.010,                 # (deprecated) kept for backward compatibility
    colorbar_width=0.020,               # (deprecated) kept for backward compatibility
    show_panel_labels=True,
    panel_labels=("(a)", "(b)", "(c)", "(d)", "(e)"),

    # ranges
    xlim=(4.0, 28.0),
    r_ylim=(0.0, 1200.0),
    v_ylim=(0.0, 60.0),
    age_ylim=(0.0, 35.0),
    n_ylim=None,
    sn_ylim=None,                       # None = auto from data (log axis, with padding)

    # colormap / colorbar for n_HI_ring_cm-3
    n_vmin=None,
    n_vmax=None,
    bubble_cmap_name="coolwarm",
    colorbar_label=r"$n_{\rm HI}$ [cm$^{-3}$]",

    # bubble draw parameters
    bubble_mode="circle",   # 'rect' or 'circle'

    # rectangle style
    bubble_rect_face_alpha=0.15,
    bubble_rect_edge_alpha=0.95,
    bubble_rect_edge_lw=0.15,
    bubble_rect_zorder=5,

    # circle style
    bubble_marker_size=45.0,
    bubble_marker_face_alpha=0.70,
    bubble_use_data_edgecolor=False,
    bubble_marker_edge_color=(0.0, 0.0, 0.0, 1.0),
    bubble_marker_edge_alpha=0.95,
    bubble_marker_edge_lw=0.4,
    bubble_marker_zorder=6,

    # errorbars
    bubble_show_errorbar=True,
    bubble_errorbar_use_data_color=True,
    bubble_errorbar_data_alpha=0.80,
    bubble_errorbar_color=(0.9, 0.4, 0.4, 0.25),
    bubble_errorbar_elinewidth=0.6,
    bubble_errorbar_capsize=0.0,
    bubble_errorbar_linestyle='-',
    bubble_errorbar_zorder=5,

    # per-panel y-errors (choose: 'none' | 'fractional' | 'abs')
    r_yerr_mode="fractional",   r_yerr_value=0.20,
    v_yerr_mode="fractional",   v_yerr_value=0.10,
    age_yerr_mode="fractional", age_yerr_value=0.20,
    n_yerr_mode="fractional",   n_yerr_value=0.20,   # panel d: fractional y-error, default ±20%
    sn_yerr_mode="fractional",  sn_yerr_value=0.50,  # panel e: fractional y-error, default ±50%

    # output
    output_pdf="bubble_r_v_age_nsn_vs_r.pdf",
):
    # base style
    plt.rcParams.update({
        "figure.dpi": 75, "savefig.dpi": 75,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10
    })

    # data
    D = _prep_bubble_data(bubble_table_path, r_fits_path, alpha_fits_path, mom0_fits_path)
    R        = D["R_kpc"]
    r_pc     = D["r_pc"]
    v_kms    = D["v_kms"]
    age      = D["age_myr"]
    xerr     = D["xerr_kpc"]
    n_ring   = D["n_ring"]
    NSN      = D["NSN_weaver"]
    h_pc     = D["h_pc"]
    angles   = D["angles"]

    # Colormap parameters are kept for backward compatibility but are not used
    # (markers are fixed red; no colorbar).

    # figure and axes by absolute boxes
    if len(bottoms) != 5:
        raise ValueError("bottoms must have length 5 for panels (a)-(e).")
    if len(panel_labels) != 5:
        raise ValueError("panel_labels must have length 5 for panels (a)-(e).")

    fig = plt.figure(figsize=fig_size)
    ax_r  = fig.add_axes([left, bottoms[0], width, panel_height])
    ax_v  = fig.add_axes([left, bottoms[1], width, panel_height])
    ax_a  = fig.add_axes([left, bottoms[2], width, panel_height])
    ax_n  = fig.add_axes([left, bottoms[3], width, panel_height])
    ax_sn = fig.add_axes([left, bottoms[4], width, panel_height])

    # cosmetics
    def style_ax(ax):
        ax.tick_params(axis='both', direction='in', length=5, top=True, right=True)
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    for ax in (ax_r, ax_v, ax_a, ax_n, ax_sn):
        style_ax(ax)
        ax.set_xlim(xlim)

    # labels
    ax_r.set_ylabel(r"$r$ [pc]")
    ax_v.set_ylabel(r"$v_{\rm exp}$ [km s$^{-1}$]")
    ax_a.set_ylabel(r"age [Myr]")
    ax_n.set_ylabel(r"$n_{\rm HI}$ [cm$^{-3}$]")
    ax_sn.set_ylabel(r"$N_{\rm SN}$")
    ax_sn.set_xlabel(r"Distance to M31's center: $R$ [kpc]")

    # Only bottom panel shows x tick labels
    ax_r.tick_params(labelbottom=False, labelleft=True)
    ax_v.tick_params(labelbottom=False, labelleft=True)
    ax_a.tick_params(labelbottom=False, labelleft=True)
    ax_n.tick_params(labelbottom=False, labelleft=True)
    ax_sn.tick_params(labelbottom=True,  labelleft=True)

    # panel tags
    if show_panel_labels:
        for ax, tag in zip((ax_r, ax_v, ax_a, ax_n, ax_sn), panel_labels):
            ax.text(0.015, 0.95, tag, transform=ax.transAxes,
                    va="top", ha="left", fontsize=10, weight="bold")

    # y ranges
    ax_r.set_ylim(r_ylim)
    ax_v.set_ylim(v_ylim)
    ax_a.set_ylim(age_ylim)

    # panel (d): n_HI in log scale
    ax_n.set_yscale('log')
    if n_ylim is not None:
        ax_n.set_ylim(n_ylim)
    else:
        finite_n = n_ring[np.isfinite(n_ring) & (n_ring > 0)]
        if finite_n.size > 0:
            y0 = float(np.nanmin(finite_n))
            y1 = float(np.nanmax(finite_n))
            ax_n.set_ylim(y0 / 1.8, y1 * 1.8)

    # panel (e): N_SN in log scale
    ax_sn.set_yscale('log')

    # build y-errors
    def build_yerr(y, mode, val):
        if mode is None or str(mode).lower() == "none":
            return None
        mode = str(mode).lower()
        if mode == "fractional":
            return np.abs(val) * y
        if mode == "abs":
            return np.full_like(y, float(val))
        return None

    yerr_r   = build_yerr(r_pc,   r_yerr_mode,   r_yerr_value)
    yerr_v   = build_yerr(v_kms,  v_yerr_mode,   v_yerr_value)
    yerr_age = build_yerr(age,    age_yerr_mode, age_yerr_value)
    yerr_n   = build_yerr(n_ring, n_yerr_mode,   n_yerr_value)
    yerr_sn  = build_yerr(NSN,    sn_yerr_mode,  sn_yerr_value)

    # keep log-y panels positive under error bars
    if yerr_n is not None:
        yerr_n = np.where(np.isfinite(n_ring) & (n_ring > 0), np.minimum(yerr_n, 0.9 * n_ring), np.nan)
    if yerr_sn is not None:
        yerr_sn = np.where(np.isfinite(NSN) & (NSN > 0), np.minimum(yerr_sn, 0.9 * NSN), np.nan)

    if bubble_mode != "circle":
        warnings.warn("bubble_mode != 'circle' is deprecated; using circle markers.", stacklevel=2)

    for ax, y, yerr in (
        (ax_r,  r_pc,   yerr_r),
        (ax_v,  v_kms,  yerr_v),
        (ax_a,  age,    yerr_age),
        (ax_n,  np.where(n_ring > 0, n_ring, np.nan), yerr_n),
        (ax_sn, np.where(NSN > 0, NSN, np.nan),       yerr_sn),
    ):
        _draw_bubbles_fixed_red(
            ax,
            R,
            y,
            xerr=xerr,
            yerr=yerr,
            marker_size=bubble_marker_size,
            face_alpha=bubble_marker_face_alpha,
            show_errorbar=bubble_show_errorbar,
            errorbar_color=bubble_errorbar_color,
            errorbar_elinewidth=bubble_errorbar_elinewidth,
            errorbar_capsize=bubble_errorbar_capsize,
            errorbar_linestyle=bubble_errorbar_linestyle,
            marker_zorder=bubble_marker_zorder,
            errorbar_zorder=bubble_errorbar_zorder,
        )

    # panel (e) Y-limit: auto if sn_ylim is None
    if sn_ylim is not None:
        ax_sn.set_ylim(sn_ylim)
    else:
        finite_nsn = NSN[np.isfinite(NSN) & (NSN > 0)]
        if finite_nsn.size > 0:
            ymin = float(np.min(finite_nsn))
            ymax = float(np.max(finite_nsn))

            # include effect of fractional y-error if used
            if sn_yerr_mode is not None and str(sn_yerr_mode).lower() == "fractional":
                frac = float(sn_yerr_value)
                if frac > 0.0:
                    ymin = ymin * max(1.0 - frac, 0.1)
                    ymax = ymax * (1.0 + frac)

            # padding in log space
            ymin *= 0.8
            ymax *= 1.2

            if ymin <= 0.0:
                ymin = ymax * 1.0e-2

            ax_sn.set_ylim(ymin, ymax)

    plt.savefig(output_pdf, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_pdf}")
    plt.close(fig)


# -----------------------
# Example call
# -----------------------
if __name__ == "__main__":
    plot_bubble_r_v_age_aligned(
        bubble_table_path="../code/1113.tab",
        r_fits_path="../data/r-m31-jcomb_modHeader.fits",
        alpha_fits_path="../data/alpha-m31-jcomb_modHeader.fits",
        mom0_fits_path="../data/jcomb_vcube_scaled2_mom0.fits",

        # layout control
        fig_size=(6.1, 9.4),
        left=0.12,
        width=0.78,
        panel_height=0.155,
        bottoms=(0.80, 0.62, 0.44, 0.26, 0.08),
        panel_labels=("(a)", "(b)", "(c)", "(d)", "(e)"),

        # ranges
        xlim=(4.0, 28.0),
        r_ylim=(0.0, 1200.0),
        v_ylim=(0.0, 38.0),
        age_ylim=(0.0, 45.0),
        n_ylim=None,
        sn_ylim=None,   # auto for panel (e)

        # bubble drawing
        bubble_mode="circle",
        bubble_marker_size=45.0,
        bubble_marker_face_alpha=0.70,
        bubble_marker_zorder=6,
        bubble_show_errorbar=True,
        bubble_errorbar_elinewidth=0.6,
        bubble_errorbar_capsize=0.0,
        bubble_errorbar_linestyle='-',
        bubble_errorbar_zorder=5,

        # per-panel y-errors
        r_yerr_mode="fractional",   r_yerr_value=0.20,
        v_yerr_mode="fractional",   v_yerr_value=0.20,
        age_yerr_mode="fractional", age_yerr_value=0.30,
        n_yerr_mode="fractional",   n_yerr_value=0.20,
        sn_yerr_mode="fractional",  sn_yerr_value=0.5,

        output_pdf="bubble_r_v_age_nsn_vs_r.pdf",
    )
