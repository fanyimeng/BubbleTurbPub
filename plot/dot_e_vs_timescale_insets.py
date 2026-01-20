#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble dotE vs timescale (t_exp) with Brinks+86 comparison crosses and an inset showing the full Brinks sample.

Bubble definitions:
- dotE_bubble = 0.5 * rho * v^3 / (h_pc * pc2cm) * 1e28 (same as dot_e_balance).
- t_exp,bub = 0.6 * r_pc / v_exp  (Myr), using expansion_vel column (+1 km/s offset, consistent with dot_e_balance).

Brinks+86:
- dotE_brinks recomputed as 0.5 * M * v^2 (M from Mass_1e4Msun, v = |DV_kms|) divided by (volume from Diam_pc) and Age_Myr.
- t_exp,brinks = Age_Myr.

Inputs
------
- bubble_table_path : ../code/1113.tab
- ring_table_path   : ../code/1113.tab (for n_HI_ring_cm-3 used in bubble dotE)
- alpha_fits        : ../data/alpha-m31-jcomb_modHeader.fits
- r_fits            : ../data/r-m31-jcomb_modHeader.fits  (for R_kpc coloring)
- brinks_table_path : ../data/brinks+86/brinks86_combined.fwf

Output
------
- dot_e_vs_timescale_insets.pdf (by default)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

pc2cm = 3e18
SEC_PER_MYR = 1e6 * 3.154e7
COSI = np.cos(np.deg2rad(77.0))

FIG_DPI = 100
SAVEFIG_DPI = 100
AX_LABELSIZE = 10
AX_TITLESIZE = 10
TICK_LABELSIZE = 10
TICK_LENGTH_MAIN = 5
TICK_LENGTH_INSET = 3
COLORBAR_LABELSIZE = 9
COLORBAR_TICKSIZE = 8
COLORBAR_PAD = 0.01
COLORBAR_ASPECT = 40
LEGEND_LOC = "lower left"
LEGEND_FONTSIZE = 7
LEGEND_FRAMEALPHA = 0.9
LEGEND_EDGE_COLOR = "black"
LEGEND_EDGE_LW = 0.5
LEGEND_ORDER = ("bubbles", "brinks", "models")
FIGSIZE_MAIN = (4.8, 4.8)
INSET_SIZE = "36%"
INSET_BORDERPAD = 0.8
INSET_TEXT_POS = (0.24, 0.96)
INSET_TEXT_FONTSIZE = 7
X_LABEL_MAIN = r"$t_{\rm exp}$ [Myr]"
Y_LABEL_MAIN = r"$\dot{e}_{\rm bubble}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]"

BUBBLE_MARKER_SIZE = 65
BUBBLE_MARKER = "o"
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)  # match rv diagram red
BUBBLE_FACE_ALPHA = 0.7
BUBBLE_EDGE_ALPHA = 0
BUBBLE_EDGE_LW = 0
BUBBLE_EDGE_COLOR = (0.9, 0.2, 0.2)

BRINKS_MARKER_SIZE = 45
BRINKS_MARKER = "^"
BRINKS_MARKER_COLOR = (0.6, 0.6, 0.9)
BRINKS_MARKER_LW = 0
BRINKS_FACE_COLOR = None  # if None, uses BRINKS_MARKER_COLOR
BRINKS_FACE_ALPHA = 0.0
BRINKS_EDGE_ALPHA = 0.6
BRINKS_EDGE_COLOR = (0.6, 0.6, 0.9)
BRINKS_ENERGY_SCALE = 0.44
LEGEND_LABEL_BRINKS = "Brinks+86"
MODEL_LABEL_TEMPLATE = r"model ($n_0={n0}$ cm$^{{-3}}$)"

MODEL_R0_LIST = (50.0, 50.0)
MODEL_T0_LIST = (1.0, 1.0)
MODEL_N0_LIST = (0.05, 0.3)
MODEL_COLORS = ("k", "0.3")
MODEL_LS = ("--", "-.")
MODEL_LW = 0.9
MODEL_ALPHA = 1.0

MODEL_R0_LIST = (50.0, 100.0)
MODEL_T0_LIST = (1.0, 1.0)
MODEL_N0_LIST = (0.03, 0.5)
MODEL_COLORS = ("gray", "0.3")
MODEL_LS = ("--", "-.")
MODEL_LW = 0.9
MODEL_ALPHA = 0.7
MODEL_LABEL_1 = r"model ($n_0={n0}$ cm$^{{-3}}$)"
MODEL_LABEL_2 = r"model ($n_0={n0}$ cm$^{{-3}}$)"
LEGEND_LABEL_BUBBLES = "New Bubbles"
LEGEND_LABEL_BRINKS = "Brinks+86"
LEGEND_ORDER = ("bubbles",  "brinks", "models")
VERT_LINE_X = 32.0
VERT_LINE_TEXT = "SNe End"
VERT_LINE_STYLE = {"color": "gray", "lw": 1.2, "ls": ":"}
VERT_LINE_TEXT_POS = (0.74, 0.26)


def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def load_bubbles(
    bubble_table_path,
    ring_table_path,
    alpha_fits,
    r_fits,
    nring_column="n_HI_ring_cm-3",
):
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
    r_bub_pc = pd.to_numeric(df_bubble["radius_pc"], errors="coerce").to_numpy(dtype=float)
    vexp_kms = pd.to_numeric(df_bubble["expansion_vel"], errors="coerce").to_numpy(dtype=float) + 1.0

    x_pos_pc = get_fits_at_pix(r_fits, ra_pix, dec_pix)
    x_pos_kpc = x_pos_pc / 1e3

    nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)

    h_pc = 182 + 16 * x_pos_kpc
    rho = 1.673e-24 * nring
    v_cms = vexp_kms * 1e5
    dotE = 0.5 * rho * v_cms**3 / (h_pc * pc2cm) * 1e28  # 10^-28 units
    t_exp_myr = 0.6 * r_bub_pc / vexp_kms  # Myr

    return {
        "R_kpc": x_pos_kpc,
        "dotE": dotE,
        "t_exp_myr": t_exp_myr,
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
    vmin = vmin if vmin > 0 else np.nanmin(v) / (pad * 2.0)
    return (vmin, vmax)


def _add_brinks_insets(
    ax,
    xvals,
    yvals,
    ylim,
    marker_size,
    marker_style,
    marker_color,
    marker_face_color,
    marker_lw,
    marker_face_alpha,
    marker_edge_alpha,
    marker_edge_color,
    tick_length_inset,
    inset_text_fontsize,
    inset_size,
    inset_borderpad,
    inset_text_pos,
):
    if xvals is None or yvals is None:
        return
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    mask_finite = np.isfinite(xvals) & np.isfinite(yvals) & (xvals > 0) & (yvals > 0)
    if not np.any(mask_finite) or ylim is None:
        return

    limits_x = _log_limits_around(xvals[mask_finite], pad=1.8)
    limits_y = _log_limits_around(yvals[mask_finite], pad=3.5)
    if limits_x is None or limits_y is None:
        return

    iax = inset_axes(ax, width=inset_size, height=inset_size, loc="upper right", borderpad=inset_borderpad)
    iax.set_xscale("log")
    iax.set_yscale("log")
    edge_base = marker_color if marker_edge_color is None else marker_edge_color
    edge_rgba = mcolors.to_rgba(edge_base, alpha=marker_edge_alpha)
    iax.scatter(
        xvals[mask_finite],
        yvals[mask_finite],
        marker=marker_style,
        s=marker_size * 0.9,
        facecolor=edge_rgba,
        edgecolor=edge_rgba,
        linewidths=max(marker_lw, 0.8),
        zorder=12,
    )
    iax.set_xlim(limits_x)
    iax.set_ylim(limits_y)
    iax.yaxis.set_minor_locator(mticker.NullLocator())
    iax.tick_params(direction="in", which="both", length=tick_length_inset, labelsize=inset_text_fontsize, top=True, right=True)
    iax.text(
        inset_text_pos[0],
        inset_text_pos[1],
        "Brinks+86 (full)",
        ha="left",
        va="top",
        fontsize=inset_text_fontsize,
        transform=iax.transAxes,
    )


def _broadcast_list(values, n):
    vals = list(values) if isinstance(values, (list, tuple, np.ndarray)) else [values]
    if len(vals) == 0:
        return [np.nan] * n
    if len(vals) >= n:
        return vals[:n]
    return vals + [vals[-1]] * (n - len(vals))


def plot_dot_e_vs_timescale(
    # --- figure layout ---
    figsize=FIGSIZE_MAIN,
    inset_size=INSET_SIZE,
    inset_borderpad=INSET_BORDERPAD,
    inset_text_pos=INSET_TEXT_POS,
    inset_text_fontsize=INSET_TEXT_FONTSIZE,
    tick_length_inset=TICK_LENGTH_INSET,
    xlim=[1,100],
    ylim=[5e-2,1e4],
    bubble_table_path="../code/1113.tab",
    ring_table_path="../code/1113.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",
    nring_column="n_HI_ring_cm-3",
    # coloring
    color_by_radius=False,
    cmap_name="coolwarm",
    color_vmin=None,
    color_vmax=None,
    colorbar_label=r"$R$ [kpc]",
    # bubble markers
    bubble_marker_size=BUBBLE_MARKER_SIZE,
    bubble_marker=BUBBLE_MARKER,
    bubble_base_face_color=BUBBLE_BASE_FACE_COLOR,
    bubble_marker_face_alpha=BUBBLE_FACE_ALPHA,
    bubble_marker_edge_alpha=BUBBLE_EDGE_ALPHA,
    bubble_marker_edge_lw=BUBBLE_EDGE_LW,
    bubble_marker_edge_color=BUBBLE_EDGE_COLOR,
    # Brinks overlay
    brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
    brinks_show=True,
    brinks_marker_size=BRINKS_MARKER_SIZE,
    brinks_marker=BRINKS_MARKER,
    brinks_face_color=BRINKS_FACE_COLOR,
    brinks_marker_color=BRINKS_MARKER_COLOR,
    brinks_marker_lw=BRINKS_MARKER_LW,
    brinks_marker_face_alpha=BRINKS_FACE_ALPHA,
    brinks_marker_edge_alpha=BRINKS_EDGE_ALPHA,
    brinks_marker_edge_color=BRINKS_EDGE_COLOR,
    # Model curves r ~ t^{3/5}
    model_show=True,
    model_r0_list=MODEL_R0_LIST,
    model_t0_list=MODEL_T0_LIST,
    model_n0_list=MODEL_N0_LIST,
    model_colors=MODEL_COLORS,
    model_ls=MODEL_LS,
    model_lw=MODEL_LW,
    # export
    export_table=False,
    export_path="dot_e_vs_timescale.csv",
    # output
    output_pdf="dot_e_vs_timescale_insets.pdf",
    # inset settings
    brinks_insets=True,
    # fonts and ticks
    fig_dpi=FIG_DPI,
    savefig_dpi=SAVEFIG_DPI,
    axes_labelsize=AX_LABELSIZE,
    axes_titlesize=AX_TITLESIZE,
    tick_labelsize=TICK_LABELSIZE,
    tick_length_main=TICK_LENGTH_MAIN,
    colorbar_labelsize=COLORBAR_LABELSIZE,
    colorbar_ticksize=COLORBAR_TICKSIZE,
    colorbar_pad=COLORBAR_PAD,
    colorbar_aspect=COLORBAR_ASPECT,
    legend_loc=LEGEND_LOC,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_framealpha=LEGEND_FRAMEALPHA,
    legend_edgecolor=LEGEND_EDGE_COLOR,
    legend_edge_lw=LEGEND_EDGE_LW,
    # model line styling
    model_alpha=MODEL_ALPHA,
):
    plt.rcParams.update({
        "figure.dpi": fig_dpi,
        "savefig.dpi": savefig_dpi,
        "axes.labelsize": axes_labelsize,
        "axes.titlesize": axes_titlesize,
        "xtick.labelsize": tick_labelsize,
        "ytick.labelsize": tick_labelsize,
    })
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL_MAIN)
    ax.set_ylabel(Y_LABEL_MAIN)
    ax.text(0.02, 0.98, "d", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.tick_params(direction="in", which="both", length=tick_length_main, top=True, right=True)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))

    bubble = load_bubbles(
        bubble_table_path=bubble_table_path,
        ring_table_path=ring_table_path,
        alpha_fits=alpha_fits,
        r_fits=r_fits,
        nring_column=nring_column,
    )

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
        facecolors = np.stack([np.r_[c[:3], bubble_marker_face_alpha] for c in colors])
        edgecolors = None
    else:
        norm = None
        face_base = plt.get_cmap(cmap_name)(0.6)[:3] if bubble_base_face_color is None else mcolors.to_rgba(bubble_base_face_color)[:3]
        base_face = np.r_[face_base, bubble_marker_face_alpha]
        facecolors = np.repeat(base_face[None, :], len(r_kpc_vals), axis=0)
        edgecolors = None

    mask = np.isfinite(bubble["t_exp_myr"]) & np.isfinite(bubble["dotE"])
    ax.scatter(
        bubble["t_exp_myr"][mask],
        bubble["dotE"][mask],
        s=bubble_marker_size,
        marker=bubble_marker,
        facecolor=facecolors[mask],
        edgecolor="none",
        linewidths=0,
        zorder=5,
    )

    brinks_handle = None
    brinks_age_myr = None
    brinks_dote_scaled = None
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
                age_s = age_myr * SEC_PER_MYR
                mass_g = mass_1e4 * 1e4 * 1.989e33
                v_cms = dv_kms * 1e5
                energy_erg = 0.5 * mass_g * v_cms**2 * BRINKS_ENERGY_SCALE
                with np.errstate(divide="ignore", invalid="ignore"):
                    dote_phys = energy_erg / (volume_cm3 * age_s)
                dote_scaled = dote_phys * 1e28

                mb = np.isfinite(age_myr) & np.isfinite(dote_scaled)
                brinks_age_myr = age_myr
                brinks_dote_scaled = dote_scaled
                if np.any(mb):
                    edge_base = brinks_marker_edge_color if brinks_marker_edge_color is not None else brinks_marker_color
                    edge_rgba = mcolors.to_rgba(edge_base, alpha=brinks_marker_edge_alpha)
                    brinks_handle = ax.scatter(
                        age_myr[mb],
                        dote_scaled[mb],
                        marker=brinks_marker,
                        s=brinks_marker_size,
                        facecolor=edge_rgba,
                        edgecolor=edge_rgba,
                        linewidths=max(brinks_marker_lw, 0.8),
                        zorder=12,
                        label=LEGEND_LABEL_BRINKS,
                    )
        else:
            print(f"[DotEvsTimescale] Brinks table not found at {bpath}, skipping crosses.")

    # Model curves: r = r0 (t/t0)^{3/5}, v = (3/5) r / t, dotE = 0.5 rho v^2 / t (per-volume rate)
    if model_show:
        t_model = np.logspace(-1, 2, 200)  # Myr
        n0_list = list(model_n0_list) if isinstance(model_n0_list, (list, tuple, np.ndarray)) else [model_n0_list]
        n_models = len(n0_list)
        r0_list = _broadcast_list(model_r0_list, n_models)
        t0_list = _broadcast_list(model_t0_list, n_models)
        colors_list = _broadcast_list(model_colors, n_models)
        ls_list = _broadcast_list(model_ls, n_models)
        t_sec_model = t_model * SEC_PER_MYR
        model_handles = []
        for idx, (n0, r0, t0, col, ls) in enumerate(zip(n0_list, r0_list, t0_list, colors_list, ls_list)):
            r_pc_model = r0 * (t_model / t0) ** (3.0 / 5.0)
            v_cms_model = (3.0 / 5.0) * r_pc_model * pc2cm / t_sec_model
            rho_model = 1.673e-24 * n0
            dotE_model = 0.5 * rho_model * v_cms_model**2 / t_sec_model * 1e28  # 10^-28 units
            label_model = MODEL_LABEL_1.format(n0=n0) if idx == 0 else MODEL_LABEL_2.format(n0=n0)
            line = ax.plot(
                t_model, dotE_model,
                color=col, ls=ls, lw=model_lw, alpha=model_alpha,
                label=label_model,
                zorder=4,
            )[0]
            model_handles.append(line)

    if color_by_radius:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=colorbar_pad, aspect=colorbar_aspect)
        cbar.set_label(colorbar_label, fontsize=colorbar_labelsize)
        cbar.ax.tick_params(labelsize=colorbar_ticksize)

    if VERT_LINE_X is not None:
        ax.axvline(VERT_LINE_X, **VERT_LINE_STYLE)
        if VERT_LINE_TEXT:
            ax.text(
                VERT_LINE_TEXT_POS[0],
                VERT_LINE_TEXT_POS[1],
                VERT_LINE_TEXT,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=legend_fontsize,
                rotation=90,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
            )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)

    if brinks_insets:
        _add_brinks_insets(
            ax=ax,
            xvals=brinks_age_myr,
            yvals=brinks_dote_scaled,
            ylim=ax.get_ylim(),
            marker_size=brinks_marker_size,
            marker_style=brinks_marker,
            marker_color=brinks_marker_color,
            marker_face_color=brinks_face_color,
            marker_lw=brinks_marker_lw,
            marker_face_alpha=brinks_marker_face_alpha,
            marker_edge_alpha=brinks_marker_edge_alpha,
            marker_edge_color=brinks_marker_edge_color,
            tick_length_inset=tick_length_inset,
            inset_text_fontsize=inset_text_fontsize,
            inset_size=inset_size,
            inset_borderpad=inset_borderpad,
            inset_text_pos=inset_text_pos,
        )

    final_handles = []
    final_labels = []
    bubble_sample = Line2D(
        [], [], marker=bubble_marker, linestyle="None",
        markersize=np.sqrt(bubble_marker_size),
        markerfacecolor=mcolors.to_rgba(BUBBLE_BASE_FACE_COLOR),
        markeredgecolor=mcolors.to_rgba(BUBBLE_EDGE_COLOR),
        markeredgewidth=BUBBLE_EDGE_LW,
        label=LEGEND_LABEL_BUBBLES,
    )
    final_handles.append(bubble_sample)
    final_labels.append(LEGEND_LABEL_BUBBLES)

    # Apply legend order preference
    if model_show:
        for h in model_handles:
            final_handles.append(h)
            final_labels.append(h.get_label())
    if brinks_handle is not None:
        final_handles.append(brinks_handle)
        final_labels.append(LEGEND_LABEL_BRINKS)

    if LEGEND_ORDER:
        ordered = []
        label_map = {
            "bubbles": LEGEND_LABEL_BUBBLES,
            "models": [h.get_label() for h in model_handles] if model_show else [],
            "brinks": LEGEND_LABEL_BRINKS,
        }
        for key in LEGEND_ORDER:
            target = label_map.get(key)
            if isinstance(target, list):
                for lbl in target:
                    if lbl in final_labels:
                        i = final_labels.index(lbl)
                        ordered.append((final_handles[i], final_labels[i]))
            else:
                lbl = target
                if lbl in final_labels:
                    i = final_labels.index(lbl)
                    ordered.append((final_handles[i], final_labels[i]))
        # append any remaining not captured
        for h, lbl in zip(final_handles, final_labels):
            if (h, lbl) not in ordered:
                ordered.append((h, lbl))
        final_handles = [h for h, _ in ordered]
        final_labels = [lbl for _, lbl in ordered]
    leg = ax.legend(
        final_handles,
        final_labels,
        loc=legend_loc,
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=legend_framealpha,
        edgecolor=legend_edgecolor,
        bbox_to_anchor=(0.05, 0.05, 1, 1),
    )
    leg.get_frame().set_linewidth(legend_edge_lw)

    if export_table:
        export_df = pd.DataFrame({
            "R_kpc": bubble["R_kpc"],
            "t_exp_Myr": bubble["t_exp_myr"],
            "dotE_1e-28": bubble["dotE"],
            "nring_cm-3": bubble["nring"],
        })
        export_path = Path(export_path)
        export_df.to_csv(export_path, index=False)
        print(f"[DotEvsTimescale] Exported table to {export_path}")

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    plot_dot_e_vs_timescale()
