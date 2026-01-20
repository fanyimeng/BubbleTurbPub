#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble dotE vs timescale (t_exp) with Brinks+86 comparison crosses.

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
- dot_e_vs_timescale.pdf (by default)
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
SEC_PER_MYR = 1e6 * 3.154e7
COSI = np.cos(np.deg2rad(77.0))


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


def plot_dot_e_vs_timescale(
    xlim=[1,60],
    ylim=[5e-2,1e3],
    bubble_table_path="../code/1113.tab",
    ring_table_path="../code/1113.tab",
    alpha_fits="../data/alpha-m31-jcomb_modHeader.fits",
    r_fits="../data/r-m31-jcomb_modHeader.fits",
    nring_column="n_HI_ring_cm-3",
    # coloring
    cmap_name="coolwarm",
    color_vmin=None,
    color_vmax=None,
    colorbar_label=r"$R$ [kpc]",
    # bubble markers
    bubble_marker_size=45,
    bubble_marker_face_alpha=0.7,
    bubble_marker_edge_alpha=0.95,
    bubble_marker_edge_lw=0.4,
    bubble_marker_edge_color=(0, 0, 0, 1.0),
    # Brinks overlay
    brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
    brinks_show=True,
    brinks_marker_size=25,
    brinks_marker_color="0.2",
    # Model curves r ~ t^{3/5}
    model_show=True,
    model_r0_pc=100.0,
    model_t0_myr=1.0,
    model_n0_list=(0.01, 0.3),
    model_colors=("k", "0.3"),
    model_ls=("--", "-."),
    model_lw=0.9,
    # export
    export_table=False,
    export_path="dot_e_vs_timescale.csv",
    # output
    output_pdf="dot_e_vs_timescale.pdf",
):
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_{\rm exp}$ [Myr]")
    ax.set_ylabel(r"$\dot{e}_{\rm bubble}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]")
    ax.tick_params(direction="in", which="both", length=5, top=True, right=True)
    # ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    bubble = load_bubbles(
        bubble_table_path=bubble_table_path,
        ring_table_path=ring_table_path,
        alpha_fits=alpha_fits,
        r_fits=r_fits,
        nring_column=nring_column,
    )

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

    mask = np.isfinite(bubble["t_exp_myr"]) & np.isfinite(bubble["dotE"])
    ax.scatter(
        bubble["t_exp_myr"][mask],
        bubble["dotE"][mask],
        s=bubble_marker_size,
        facecolor=(0.4,0.4,0.8,0.7),
        edgecolor=(0.4,0.4,0.8,0.7),
        linewidths=bubble_marker_edge_lw,
        zorder=5,
    )

    brinks_handle = None
    if brinks_show and brinks_table_path is not None:
        bpath = Path(brinks_table_path)
        if 0:
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
                energy_erg = 0.5 * mass_g * v_cms**2
                with np.errstate(divide="ignore", invalid="ignore"):
                    dote_phys = energy_erg / (volume_cm3 * age_s)
                dote_scaled = dote_phys * 1e28

                mb = np.isfinite(age_myr) & np.isfinite(dote_scaled)
                if np.any(mb):
                    brinks_handle = ax.scatter(
                        age_myr[mb],
                        dote_scaled[mb],
                        marker="x",
                        s=brinks_marker_size,
                        color=brinks_marker_color,
                        linewidths=0.8,
                        zorder=6,
                        label="Brinks+86",
                    )
        else:
            print(f"[DotEvsTimescale] Brinks table not found at {bpath}, skipping crosses.")

    # Model curves: r = r0 (t/t0)^{3/5}, v = (3/5) r / t, dotE = 0.5 rho v^2 / t (per-volume rate)
    if model_show:
        t_model = np.logspace(-1, 2, 200)  # Myr
        r_pc_model = model_r0_pc * (t_model / model_t0_myr) ** (3.0 / 5.0)
        t_sec_model = t_model * SEC_PER_MYR
        v_cms_model = (3.0 / 5.0) * r_pc_model * pc2cm / t_sec_model
        for n0, col, ls in zip(model_n0_list, model_colors, model_ls):
            rho_model = 1.673e-24 * n0
            dotE_model = 0.5 * rho_model * v_cms_model**2 / t_sec_model * 1e28  # 10^-28 units
            ax.plot(
                t_model, dotE_model,
                color=col, ls=ls, lw=model_lw,
                label=fr"model ($n_0={n0}$ cm$^{{-3}}$)",
                zorder=4,
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)
    # cbar.set_label(colorbar_label, fontsize=9)
    # cbar.ax.tick_params(labelsize=8)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    handles, labels = ax.get_legend_handles_labels()
    if brinks_handle is not None and "Brinks+86" not in labels:
        handles.append(brinks_handle)
        labels.append("Brinks+86")
    leg = ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=True, framealpha=0.9, edgecolor="black")
    leg.get_frame().set_linewidth(0.5)

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
