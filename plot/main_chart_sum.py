#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy budget figure — FULL VERSION (style fully configurable)

This version uses:
- nring from '../code/1110.tab' : column 'n_HI_ring_cm-3' for dotE computation.
- SN_weaver from '../code/1110.tab' : column 'SN_weaver' for bubble coloring and stats.

Output:
- EnergyBudget_pub_ALL.pdf (or name you pass in)
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# -----------------------
# Constants and column choices
# -----------------------
pc2cm = 3e18
COSI = np.cos(np.deg2rad(77.0))

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


def finite_sorted_xy(x, y):
    """Return (x_sorted, y_sorted) with both finite and x ascending."""
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.asarray(x)[m], np.asarray(y)[m]
    if xs.size == 0:
        return xs, ys
    order = np.argsort(xs)
    return xs[order], ys[order]


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
    col_den = mom0 * 1.222e6 / 1.42 / 3600.0 * 1.823e18
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


def _auto_pick_legend_levels(total_SN, bounds, q_low=0.15, q_high=0.85):
    """
    Return two values that are far apart in color for bubble legend samples.
    Prefer data quantiles; fall back to midpoints of the first and last bins.
    """
    tsn = np.asarray(total_SN, dtype=float)
    tsn = tsn[np.isfinite(tsn)]
    bmin, bmax = np.nanmin(bounds), np.nanmax(bounds)

    if tsn.size >= 5:
        lo = np.nanpercentile(tsn, q_low * 100.0)
        hi = np.nanpercentile(tsn, q_high * 100.0)
        if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
            return (float(np.clip(lo, bmin, bmax)),
                    float(np.clip(hi, bmin, bmax)))

    if bounds.size >= 3:
        v1 = 0.5 * (bounds[0] + bounds[1])
        v2 = 0.5 * (bounds[-2] + bounds[-1])
        return (float(v1), float(v2))

    return (float(bmin), float(bmax))


# -----------------------
# Main plotting function
# -----------------------
def plot_energy_budget(
    # axes limits
    xlim_full=(4.0, 28.0),
    ylim=(2e-3, 2e2),

    # turb band input and style
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

    # bubbles input
    bubble_table_path="../code/0709-decon_hb.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",
    mom0_fits="../data/jcomb_vcube_scaled2_mom0.fits",  # kept but not used for n now

    # ring table (nring and SN_weaver)
    ring_table_path="../code/1110.tab",
    nring_column="n_HI_ring_cm-3",
    snweaver_column="SN_weaver",

    # bubble color map and colorbar
    bubble_bounds=np.array([0, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200]),
    bubble_cmap_name="coolwarm",
    colorbar_label="$N_{\\rm SN}$ per bubble",
    colorbar_aspect=40,

    # bubble legend sample levels
    bubble_legend_levels=None,

    # bubble drawing and styles
    bubble_mode="circle",   # 'rect' or 'circle'
    # rectangle styles
    bubble_rect_face_alpha=0.15,
    bubble_rect_edge_alpha=0.95,
    bubble_rect_edge_lw=0.15,
    bubble_rect_zorder=5,
    # circle styles
    bubble_marker_size=45,
    bubble_marker_face_alpha=0.30,
    bubble_use_data_edgecolor=True,
    bubble_marker_edge_color=(0, 0, 0, 1.0),
    bubble_marker_edge_alpha=0.95,
    bubble_marker_edge_lw=0.4,
    bubble_marker_zorder=6,
    # errorbar control for circle mode
    bubble_show_errorbar=True,
    bubble_errorbar_use_data_color=True,
    bubble_errorbar_data_alpha=0.55,
    bubble_errorbar_color=(0, 0, 0, 0.55),
    bubble_errorbar_elinewidth=0.6,
    bubble_errorbar_capsize=0.0,
    bubble_errorbar_linestyle='-',
    bubble_errorbar_zorder=5,

    # legend
    show_legend=True,
    legend_loc="lower left",
    legend_fontsize=9,
    legend_frame=True,
    legend_framealpha=0.9,
    legend_edgecolor="black",

    # MRI theory (from resampled_med profile)
    theory_mode="shade",     # 'shade' or 'lines'
    theory_legend_mode="annotations",  # 'annotations' or 'legend'
    gi_profile_path="../data/profile_resampled_med.tsv",  # kept in signature but unused
    mri_profile_path="../data/profile_resampled_med.tsv",
    gi_xlim=(4.0, 28.0),     # kept in signature but unused
    mri_xlim=(4.0, 28.0),
    # shade params
    gi_shade_color=(0.9, 0.5, 0.1, 0.26),   # unused
    mri_shade_color=(0.2, 0.9, 0.2, 0.26),
    # line params
    gi_line_color=(0.3, 0.3, 0.3, 0.95),    # unused
    gi_linewidth=0.8,                       # unused
    gi_linestyle=(10, (1, 4)),              # unused
    mri_line_color=(0.3, 0.3, 0.3, 0.95),
    mri_linewidth=0.6,
    mri_linestyle="--",
    draw_theory_labels=True,
    theory_labels=None,  # only used when theory_legend_mode='annotations'

    # output
    output_pdf="EnergyBudget_pub_ALL.pdf",
):
    """
    Turbulence band from MAX/MIN resampled profiles.
    MRI from resampled_med.
    Bubble mode supports rectangle or circle with error bars.
    nring and SN_weaver are read from ring_table_path (../code/1110.tab).
    """

    # base style
    plt.rcParams.update({
        "figure.dpi": 75,
        "savefig.dpi": 75,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.set_yscale("log")
    ax.set_xlabel("Distance to M31's center: $R$ [kpc]")
    ax.set_ylabel(r"$\dot{e}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]")
    ax.tick_params(direction="in", which="both", length=5, top=True, bottom=True, left=True, right=True)
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
    vexp_kms = df_bubble["expansion_vel"].values + 1  # small offset as before

    angles = get_fits_at_pix(alpha_fits, ra_pix, dec_pix) - 90.0
    x_pos = get_fits_at_pix(r_fits, ra_pix, dec_pix) / 1e3

    # 2) read nring and SN_weaver from 1110.tab
    df_ring = pd.read_fwf(ring_table_path, header=0, infer_nrows=int(1e6))
    if nring_column not in df_ring.columns:
        raise KeyError(f"'{nring_column}' not found in {ring_table_path}")
    if snweaver_column not in df_ring.columns:
        raise KeyError(f"'{snweaver_column}' not found in {ring_table_path}")

    nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)
    total_SN = pd.to_numeric(df_ring[snweaver_column], errors="coerce").to_numpy(dtype=float)

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
    _print_minmax("N_SN (SN_weaver)", total_SN)

    # sum of SN
    try:
        sn_sum = float(np.nansum(total_SN))
    except Exception:
        sn_sum = np.nan
    print(f"[EnergyBudget] Sum of required SNe over all bubbles (Weaver, 1110): {sn_sum:.6f}")
    try:
        sn_sum_path = output_pdf.rsplit(".", 1)[0] + "_SNsum.txt"
        with open(sn_sum_path, "w") as f:
            f.write(f"{sn_sum:.6f}\n")
    except Exception as e:
        print(f"[EnergyBudget] Warning: could not write SN sum file: {e}")

    # color mapping by SN_weaver
    bounds = np.asarray(bubble_bounds, dtype=float)
    n_levels = len(bounds) - 1
    cmap = ListedColormap(plt.get_cmap(bubble_cmap_name)(np.linspace(0, 1, n_levels)))
    norm = BoundaryNorm(boundaries=bounds, ncolors=n_levels)

    legend_items = []
    legend_labels = []

    # Draw bubbles
    if bubble_mode == "rect":
        for i in range(len(x_pos)):
            h_i = 182 + 16 * x_pos[i]
            dh = h_i * np.tan(np.deg2rad(77)) * np.cos(np.deg2rad(angles[i]))
            dr = r_bub_pc[i]
            dxk = np.sqrt(dh**2 + dr**2) / 1e3
            base_color = cmap(norm(total_SN[i])) if np.isfinite(total_SN[i]) else (0.3, 0.3, 0.3, 0.2)
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
        data_colors = np.array([cmap(norm(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 0.3) for v in total_SN])

        # facecolor uses data color with configured alpha
        facecolors = np.stack([np.r_[c[:3], bubble_marker_face_alpha] for c in data_colors])

        # edgecolor: data color with configured alpha, or uniform color with alpha
        if bubble_use_data_edgecolor:
            edgecolors = np.stack([np.r_[c[:3], bubble_marker_edge_alpha] for c in data_colors])
        else:
            ec = np.r_[bubble_marker_edge_color[:3], bubble_marker_edge_alpha]
            edgecolors = np.repeat(ec[None, :], len(x_pos), axis=0)

        ax.scatter(
            x_pos, dotE,
            s=bubble_marker_size,
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

    # colorbar for SN coloring
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=colorbar_aspect, ticks=bounds)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # MRI theory from resampled_med profile
    def compute_mri_from_resampled(df_med: pd.DataFrame):
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

    df_med_mri = pd.read_csv(mri_profile_path, sep=r"\t", engine="python")
    x_mri_all, y_mri_all = compute_mri_from_resampled(df_med_mri)
    x_mri, y_mri = clip_xy(x_mri_all, y_mri_all, mri_xlim[0], mri_xlim[1])

    # set axes limits before theory draw
    ax.set_xlim(xlim_full)
    if ylim is not None:
        ax.set_ylim(ylim)
    ybase = ax.get_ylim()[0]

    theory_handles = []
    theory_labels_list = []

    if theory_mode == "shade":
        mri_handle = None
        if x_mri.size:
            mri_handle = ax.fill_between(x_mri, ybase, y_mri, color=mri_shade_color, linewidth=0)

        if theory_legend_mode == "legend":
            if mri_handle is not None:
                theory_handles.append(patches.Patch(facecolor=mri_shade_color, edgecolor='none'))
                theory_labels_list.append(r"$\mathrm{MRI}$")
        elif theory_legend_mode == "annotations" and draw_theory_labels and (theory_labels is not None):
            if "mri" in theory_labels:
                label = theory_labels["mri"]
                ax.annotate(
                    label["text"], xy=label["xy"], xytext=label["xytext"],
                    textcoords="data", fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="black")
                )

    elif theory_mode == "lines":
        mri_hdl = None
        if x_mri.size:
            mri_hdl, = ax.plot(x_mri, y_mri, color=mri_line_color, lw=mri_linewidth, ls=mri_linestyle)

        if theory_legend_mode == "legend":
            if mri_hdl is not None:
                theory_handles.append(Line2D([], [], color=mri_line_color, lw=mri_linewidth, ls=mri_linestyle))
                theory_labels_list.append(r"$\mathrm{MRI}$")
        elif theory_legend_mode == "annotations" and draw_theory_labels and (theory_labels is not None):
            if "mri" in theory_labels:
                label = theory_labels["mri"]
                ax.annotate(
                    label["text"], xy=label["xy"], xytext=label["xytext"],
                    textcoords="data", fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="black")
                )
    else:
        raise ValueError("theory_mode must be 'shade' or 'lines'.")

    # Legend
    if show_legend:
        if (bubble_legend_levels is None) or (isinstance(bubble_legend_levels, str) and bubble_legend_levels.lower() == "auto"):
            sample_levels = _auto_pick_legend_levels(total_SN, bounds, q_low=0.15, q_high=0.85)
        else:
            try:
                lv = tuple(bubble_legend_levels)
                sample_levels = (lv[0], lv[1])
            except Exception:
                sample_levels = _auto_pick_legend_levels(total_SN, bounds, q_low=0.15, q_high=0.85)

        sample_colors = [cmap(norm(v)) for v in sample_levels]

        if bubble_mode == "rect":
            bubble_patch_group = tuple(
                Rectangle((0, 0), 1, 1,
                          facecolor=col[:3] + (bubble_rect_face_alpha,),
                          edgecolor='k',
                          lw=bubble_rect_edge_lw,
                          alpha=None)
                for col in sample_colors
            )
            legend_items.append(bubble_patch_group)
            legend_labels.append(r"$\dot{e}_{\rm bubble}$")
        elif bubble_mode == "circle":
            circ_group = []
            for col in sample_colors:
                fc = col[:3] + (bubble_marker_face_alpha,)
                if bubble_use_data_edgecolor:
                    ec = col[:3] + (bubble_marker_edge_alpha,)
                else:
                    ec = np.r_[bubble_marker_edge_color[:3], bubble_marker_edge_alpha]
                circ = Line2D([], [], marker='o', linestyle='None',
                              markersize=np.sqrt(bubble_marker_size),
                              markerfacecolor=fc, markeredgecolor=ec, markeredgewidth=bubble_marker_edge_lw)
                circ_group.append(circ)
            bubble_marker_group = tuple(circ_group)
            legend_items.append(bubble_marker_group)
            legend_labels.append(r"$\dot{e}_{\rm bubble}$")

        if turb_band is not None:
            legend_items.append(turb_band)
            legend_labels.append(r"$\dot{e}_{\rm turb}$")

        legend_items.extend(theory_handles)
        legend_labels.extend(theory_labels_list)

        leg = ax.legend(
            legend_items, legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame
        )
        if legend_frame:
            leg.get_frame().set_edgecolor(legend_edgecolor)
            leg.get_frame().set_alpha(legend_framealpha)
            leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


# -----------------------
# Example call
# -----------------------
if __name__ == "__main__":
    plot_energy_budget(
        # axes
        xlim_full=(4, 28),
        ylim=(2e-3, 2e2),

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
        bubble_table_path="../code/0709-decon_hb.tab",
        alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
        r_fits="../data/r-m31-jcomb_modHeader.fits",
        mom0_fits="../data/jcomb_vcube_scaled2_mom0.fits",
        ring_table_path="../code/1110.tab",
        nring_column="n_HI_ring_cm-3",
        snweaver_column="SN_weaver",

        # coloring
        bubble_bounds=np.array([0, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200]),
        bubble_cmap_name="coolwarm",

        # circle mode with data colored error bars
        bubble_mode="circle",
        bubble_marker_size=45,
        bubble_marker_face_alpha=0.70,
        bubble_use_data_edgecolor=False,
        bubble_marker_edge_alpha=0.95,
        bubble_marker_edge_lw=0.4,
        bubble_marker_zorder=6,
        bubble_show_errorbar=True,
        bubble_errorbar_use_data_color=True,
        bubble_errorbar_data_alpha=0.8,
        bubble_errorbar_elinewidth=0.6,
        bubble_errorbar_capsize=0.0,
        bubble_errorbar_linestyle='-',
        bubble_errorbar_zorder=5,

        # legend
        show_legend=True,
        legend_loc="lower left",
        legend_fontsize=8,
        legend_frame=True,
        legend_framealpha=0.9,
        legend_edgecolor="black",
        bubble_legend_levels=None,

        # MRI from resampled_med
        theory_mode="shade",
        theory_legend_mode="legend",
        gi_profile_path="../data/profile_resampled_med.tsv",
        mri_profile_path="../data/profile_resampled_med.tsv",
        gi_xlim=(3, 33),
        mri_xlim=(3, 33),
        gi_line_color=(0.4, 0.4, 0.4, 0.95),
        gi_linewidth=0.8,
        gi_linestyle=(10, (1, 4)),
        mri_line_color=(0.4, 0.4, 0.4, 0.95),
        mri_linewidth=0.6,
        mri_linestyle="--",
        draw_theory_labels=True,
        theory_labels={
            "mri": {"text": r"$\rm MRI$", "xy": (23, 1e-2), "xytext": (24.5, 0.8e-2)}
        },

        # output
        output_pdf="EnergyBudget_pub_ALL_sum.pdf",
    )