#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble v^2 vs turbulent v^2 at the same radius (R_kpc).

Inputs
------
- bubble_table_path : Fixed-width bubble table (default: ../code/1113.tab).
- ring_table_path   : Ring table providing n_HI_ring_cm-3 for coloring (default: ../code/1113.tab).
- alpha_fits        : Angle map FITS (used only for alignment; not needed for v^2 itself).
- r_fits            : Deprojected radius map FITS in pc (used to locate R_kpc for interpolation).
- prof_max_path     : Resampled profile TSV (max branch) with columns ['r', x1_mom0, x1_mom2].
- prof_min_path     : Resampled profile TSV (min branch) with columns ['r', x1_mom0, x1_mom2].

Outputs
-------
- v2_vs_turb.pdf : Scatter of bubble v^2 (y) vs turbulent v^2 at same R (x), both in (km/s)^2.
- Optional CSV (export_table=True) with per-bubble values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from pathlib import Path

pc2cm = 3e18
COSI = np.cos(np.deg2rad(77.0))
TURB_MOM0_COL = "x1_mom0"
TURB_MOM2_COL = "x1_mom2"


def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def compute_turb_sigma2(df: pd.DataFrame, branch: str):
    """
    Return r_kpc, sigma_kms^2 from a resampled profile DataFrame.
    branch: 'max' -> h_pc = 182 - 37 + 13 * r_kpc
            'min' -> h_pc = 182 + 37 + 19 * r_kpc
    """
    r_pc = df["r"].to_numpy(dtype=float)
    r_kpc = r_pc / 1e3
    mom2 = df[TURB_MOM2_COL].to_numpy(dtype=float)
    v2 = np.clip(mom2**2 - 0.6 * 8.0**2, 0.0, None)  # (km/s)^2 after correction
    return r_kpc, v2


def interpolate_at_r(r_grid, val_grid, r_targets):
    return np.interp(r_targets, r_grid, val_grid, left=np.nan, right=np.nan)


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
    vexp_kms = df_bubble["expansion_vel"].to_numpy(dtype=float) + 1.0  # match dot_e_balance offset
    x_pos_pc = get_fits_at_pix(r_fits, ra_pix, dec_pix)
    x_pos_kpc = x_pos_pc / 1e3

    nring = pd.to_numeric(df_ring[nring_column], errors="coerce").to_numpy(dtype=float)
    return {
        "R_kpc": x_pos_kpc,
        "vexp_kms": vexp_kms,
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


def plot_v2_vs_turb(
    xlim=(30, 2e3),
    ylim=(30, 2e3),
    bubble_table_path="../code/1113.tab",
    ring_table_path="../code/1113.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",
    nring_column="n_HI_ring_cm-3",
    prof_max_path="../data/profile_resampled_max.tsv",
    prof_min_path="../data/profile_resampled_min.tsv",
    turb_value_mode="mean",  # 'mean', 'min', 'max'
    bubble_marker_size=45,
    bubble_marker_face_alpha=0.7,
    bubble_marker_edge_alpha=0.95,
    bubble_marker_edge_lw=0.4,
    bubble_marker_edge_color=(0, 0, 0, 1.0),
    cmap_name="coolwarm",
    color_vmin=None,
    color_vmax=None,
    colorbar_label=r"$R$ [kpc]",
    legend_loc="upper left",

    # Brinks+86 overlay
    brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
    brinks_show=True,
    brinks_marker_size=20,
    brinks_marker_color="0.2",
    export_table=False,
    export_path="v2_vs_turb.csv",
    output_pdf="v2_vs_turb.pdf",
):
    """Scatter of bubble v^2 vs turbulent v^2 in (km/s)^2."""

    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_{\rm turb}^2(R)$ [(km s$^{-1}$)$^2$]")
    ax.set_ylabel(r"$v_{\rm bubble}^2$ [(km s$^{-1}$)$^2$]")
    ax.tick_params(direction="in", which="both", length=5, top=True, right=True)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    df_max = pd.read_csv(prof_max_path, sep=r"\t", engine="python")
    df_min = pd.read_csv(prof_min_path, sep=r"\t", engine="python")
    r_kpc_max, v2_max = compute_turb_sigma2(df_max, branch="max")
    r_kpc_min, v2_min = compute_turb_sigma2(df_min, branch="min")

    bubble = load_bubbles(
        bubble_table_path=bubble_table_path,
        ring_table_path=ring_table_path,
        alpha_fits=alpha_fits,
        r_fits=r_fits,
        nring_column=nring_column,
    )

    turb_v2_max = interpolate_at_r(r_kpc_max, v2_max, bubble["R_kpc"])
    turb_v2_min = interpolate_at_r(r_kpc_min, v2_min, bubble["R_kpc"])
    if turb_value_mode == "mean":
        turb_v2 = 0.5 * (turb_v2_max + turb_v2_min)
    elif turb_value_mode == "max":
        turb_v2 = turb_v2_max
    elif turb_value_mode == "min":
        turb_v2 = turb_v2_min
    else:
        raise ValueError("turb_value_mode must be 'mean', 'min', or 'max'.")

    r_kpc_vals = bubble["R_kpc"]
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
    edgecolors = np.repeat(np.r_[bubble_marker_edge_color[:3], bubble_marker_edge_alpha][None, :], len(colors), axis=0)

    mask = np.isfinite(turb_v2) & np.isfinite(bubble["vexp_kms"])
    ax.scatter(
        turb_v2[mask],
        (bubble["vexp_kms"][mask])**2,
        s=bubble_marker_size,
        facecolor=facecolors[mask],
        edgecolor=edgecolors[mask],
        linewidths=bubble_marker_edge_lw,
        zorder=5,
    )

    line_lo = max(xlim[0], ylim[0])
    line_hi = min(xlim[1], ylim[1])
    line_x = np.array([line_lo, line_hi])
    ax.plot(line_x, line_x, color="k", lw=0.8, ls="--", zorder=4, label="1:1")

    def _sigma_log(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if not np.any(m):
            return np.nan
        d = np.log10(y[m] / x[m])
        return np.sqrt(np.nanmean(d**2))

    sigma_log = _sigma_log(turb_v2[mask], (bubble["vexp_kms"][mask])**2)
    if np.isfinite(sigma_log):
        ax.text(
            0.05, 0.95,
            rf"$\sigma_{{\log}} = {sigma_log:.2f}$ dex",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    # Brinks+86 overlay as crosses
    brinks_handle = None
    if brinks_show and brinks_table_path is not None:
        bpath = Path(brinks_table_path)
        if bpath.exists():
            bdf = pd.read_fwf(bpath)
            required = ["R_kpc", "DV_kms"]
            missing = [c for c in required if c not in bdf.columns]
            if not missing:
                r_b = pd.to_numeric(bdf["R_kpc"], errors="coerce").to_numpy(dtype=float)
                dv_b = np.abs(pd.to_numeric(bdf["DV_kms"], errors="coerce").to_numpy(dtype=float))

                turb_b_max = interpolate_at_r(r_kpc_max, v2_max, r_b)
                turb_b_min = interpolate_at_r(r_kpc_min, v2_min, r_b)
                if turb_value_mode == "mean":
                    turb_both = 0.5 * (turb_b_max + turb_b_min)
                elif turb_value_mode == "max":
                    turb_both = turb_b_max
                else:
                    turb_both = turb_b_min

                y_b = dv_b**2
                mb = np.isfinite(turb_both) & np.isfinite(y_b)
                if np.any(mb):
                    brinks_handle = ax.scatter(
                        turb_both[mb],
                        y_b[mb],
                        marker="x",
                        s=brinks_marker_size,
                        color=brinks_marker_color,
                        linewidths=0.8,
                        zorder=6,
                        label="Brinks+86",
                    )
        else:
            print(f"[V2vsTurb] Brinks table not found at {bpath}, skipping crosses.")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    leg = ax.legend(loc=legend_loc, fontsize=8, frameon=True, framealpha=0.9, edgecolor="black")
    leg.get_frame().set_linewidth(0.5)

    if export_table:
        export_df = pd.DataFrame({
            "R_kpc": bubble["R_kpc"],
            "v2_turb_kms2": turb_v2,
            "v2_bubble_kms2": (bubble["vexp_kms"])**2,
            "nring_cm-3": bubble["nring"],
        })
        export_path = Path(export_path)
        export_df.to_csv(export_path, index=False)
        print(f"[V2vsTurb] Exported table to {export_path}")

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    plot_v2_vs_turb()
