#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decompose the original density-weighted dot-e by one factor at a time."""

from __future__ import annotations

from paper_formula_common import REPO_ROOT, RUN_ROOT

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_RUN_ROOT = SCRIPT_DIR.parent
RUN_ROOT = SCRIPT_DIR.parent / "output"
REPO_ROOT = SCRIPT_DIR.parent
LOCAL_DIR = REPO_ROOT / ".local"
MPL_DIR = LOCAL_DIR / "matplotlib"
CACHE_DIR = LOCAL_DIR / "cache"
FONTCONFIG_DIR = CACHE_DIR / "fontconfig"
for directory in (MPL_DIR, CACHE_DIR, FONTCONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("FONTCONFIG_CACHE", str(FONTCONFIG_DIR))

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from paper_formula_common import (
    CORRECTED_CATALOG,
    CORRECTED_TURB_TABLE,
    MU_H_G,
    R_FITS,
    SEC_PER_MYR,
    SOURCE_BRINKS_TABLE,
    bubble_dote_1e28,
    corrected_brinks,
    expansion_time_s,
)


PC2CM = 3.0e18
M_H = MU_H_G

DEFAULT_TURB_TABLE = CORRECTED_TURB_TABLE
DEFAULT_BUBBLE_TABLE = CORRECTED_CATALOG
DEFAULT_RING_TABLE = CORRECTED_CATALOG
DEFAULT_R_FITS = R_FITS
DEFAULT_BRINKS_TABLE = SOURCE_BRINKS_TABLE
DEFAULT_OUTPUT_DIR = RUN_ROOT
DEFAULT_OUTPUT_NHI = DEFAULT_OUTPUT_DIR / "independence_edot_div_nhi.pdf"
DEFAULT_OUTPUT_V2 = DEFAULT_OUTPUT_DIR / "independence_edot_div_v2.pdf"
DEFAULT_OUTPUT_TIME = DEFAULT_OUTPUT_DIR / "independence_edot_times_time.pdf"

FIGSIZE = (4.8, 4.8)
BUBBLE_COLOR = (0.9, 0.4, 0.4)
BRINKS_COLOR = (0.6, 0.6, 0.9)
TURB_COLOR = (0.12, 0.12, 0.12, 0.35)


def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def print_minmax(name, arr):
    vals = np.asarray(arr, dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        print(f"[PanelsHI] {name}: no finite values")
        return
    print(f"[PanelsHI] {name}: min={np.nanmin(vals[mask]):.6g}, max={np.nanmax(vals[mask]):.6g}, N={np.count_nonzero(mask)}")


def bubble_rate(vexp_kms, radius_pc, n_hi):
    return bubble_dote_1e28(radius_pc, vexp_kms, n_hi)


def tdiss_myr(sigma_kms, h_kpc):
    sigma_cms = np.asarray(sigma_kms, dtype=float) * 1.0e5
    h_cm = np.asarray(h_kpc, dtype=float) * 1.0e3 * PC2CM
    return 2.0 * h_cm / sigma_cms / SEC_PER_MYR


def v2t_to_edot_per_nhi(v2t_km2_s2_myr):
    """Convert (km/s)^2/Myr to dot-e/n_HI in 10^-28 erg s^-1 H^-1."""
    return M_H * np.asarray(v2t_km2_s2_myr, dtype=float) * 1.0e10 / SEC_PER_MYR * 1.0e28


def ordered_band(y1, y2):
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    return np.fmin(y1, y2), np.fmax(y1, y2)


def load_bubbles():
    df = pd.read_fwf(DEFAULT_BUBBLE_TABLE, header=0, infer_nrows=int(1e6))
    ring = pd.read_fwf(DEFAULT_RING_TABLE, header=0, infer_nrows=int(1e6))
    if "n_HI_ring_cm-3" not in ring.columns:
        raise KeyError(f"n_HI_ring_cm-3 not found in {DEFAULT_RING_TABLE}")
    if len(df) != len(ring):
        raise ValueError(f"Bubble table length {len(df)} != ring table length {len(ring)}")
    x = get_fits_at_pix(DEFAULT_R_FITS, df["ra_pix"], df["dec_pix"]) / 1.0e3
    radius_pc = pd.to_numeric(df["radius_pc"], errors="coerce").to_numpy(dtype=float)
    vexp = pd.to_numeric(df["expansion_vel"], errors="coerce").to_numpy(dtype=float)
    n_hi = pd.to_numeric(ring["n_HI_ring_cm-3"], errors="coerce").to_numpy(dtype=float)
    rate = bubble_rate(vexp, radius_pc, n_hi)
    v2_term = vexp**2
    texp = expansion_time_s(radius_pc, vexp) / SEC_PER_MYR
    with np.errstate(divide="ignore", invalid="ignore"):
        y_div_nhi = rate / n_hi
    y_div_v2 = rate / v2_term
    y_mul_time = rate * texp
    return x, y_div_nhi, y_div_v2, y_mul_time


def load_brinks():
    if not Path(DEFAULT_BRINKS_TABLE).exists():
        return None
    df = corrected_brinks(pd.read_fwf(DEFAULT_BRINKS_TABLE))
    required = (
        "R_kpc",
        "t_exp_paper_Myr",
        "v_exp_kms",
        "nHI_updated_cm3",
        "dot_e_paper_1e-28",
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        print(f"[PanelsHI] Missing Brinks columns {missing}; skipping Brinks.")
        return None
    x = pd.to_numeric(df["R_kpc"], errors="coerce").to_numpy(dtype=float)
    age = pd.to_numeric(df["t_exp_paper_Myr"], errors="coerce").to_numpy(dtype=float)
    dv = pd.to_numeric(df["v_exp_kms"], errors="coerce").to_numpy(dtype=float)
    n_hi = pd.to_numeric(df["nHI_updated_cm3"], errors="coerce").to_numpy(dtype=float)
    rate = pd.to_numeric(df["dot_e_paper_1e-28"], errors="coerce").to_numpy(dtype=float)
    v2_term = dv**2
    with np.errstate(divide="ignore", invalid="ignore"):
        y_div_nhi = rate / n_hi
    y_div_v2 = rate / v2_term
    y_mul_time = rate * age
    return x, y_div_nhi, y_div_v2, y_mul_time


def load_turbulence_band():
    df = pd.read_csv(DEFAULT_TURB_TABLE, sep=r"\t", engine="python")
    required = (
        "r_center_kpc",
        "h_kpc",
        "sigma_nt_p16_kms",
        "sigma_nt_p84_kms",
        "dote_p16_1e-28",
        "dote_p84_1e-28",
        "v2t_p16_km2_s2_myr",
        "v2t_p84_km2_s2_myr",
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing turbulence columns in {DEFAULT_TURB_TABLE}: {missing}")
    x = pd.to_numeric(df["r_center_kpc"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["h_kpc"], errors="coerce").to_numpy(dtype=float)
    sig_lo = pd.to_numeric(df["sigma_nt_p16_kms"], errors="coerce").to_numpy(dtype=float)
    sig_hi = pd.to_numeric(df["sigma_nt_p84_kms"], errors="coerce").to_numpy(dtype=float)

    rate_lo = pd.to_numeric(df["dote_p16_1e-28"], errors="coerce").to_numpy(dtype=float)
    rate_hi = pd.to_numeric(df["dote_p84_1e-28"], errors="coerce").to_numpy(dtype=float)
    v2t_lo = pd.to_numeric(df["v2t_p16_km2_s2_myr"], errors="coerce").to_numpy(dtype=float)
    v2t_hi = pd.to_numeric(df["v2t_p84_km2_s2_myr"], errors="coerce").to_numpy(dtype=float)

    y_nhi_lo = v2t_to_edot_per_nhi(v2t_lo)
    y_nhi_hi = v2t_to_edot_per_nhi(v2t_hi)

    v2_term_lo = 3.0 * sig_lo**2
    v2_term_hi = 3.0 * sig_hi**2
    y_h_lo = rate_lo / v2_term_lo
    y_h_hi = rate_hi / v2_term_hi

    t_lo = tdiss_myr(sig_lo, h)
    t_hi = tdiss_myr(sig_hi, h)
    y_i_lo = rate_lo * t_lo
    y_i_hi = rate_hi * t_hi
    return x, y_nhi_lo, y_nhi_hi, y_h_lo, y_h_hi, y_i_lo, y_i_hi


def draw_common(ax, panel_label, ylabel, ylim):
    ax.set_yscale("log")
    ax.set_xlim(4.0, 28.0)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("Distance to M31's center: $R$ [kpc]")
    ax.set_ylabel(ylabel)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.tick_params(direction="in", which="both", length=5, top=True, bottom=True, left=True, right=True)


def scatter_points(ax, x_bub, y_bub, brinks, ylabel_for_print):
    bubble_handle = ax.scatter(
        x_bub,
        y_bub,
        s=65,
        marker="o",
        facecolors=mcolors.to_rgba(BUBBLE_COLOR, alpha=0.55),
        edgecolors=mcolors.to_rgba("white", alpha=1.0),
        linewidths=0.45,
        zorder=11,
        label="New Bubbles",
    )
    brinks_handle = None
    if brinks is not None:
        x_br, y_br, _ = brinks
        mask = np.isfinite(x_br) & np.isfinite(y_br) & (y_br > 0.0)
        brinks_handle = ax.scatter(
            x_br[mask],
            y_br[mask],
            s=55,
            marker="^",
            facecolors=mcolors.to_rgba(BRINKS_COLOR, alpha=0.55),
            edgecolors=mcolors.to_rgba("white", alpha=1.0),
            linewidths=0.45,
            zorder=10,
            label="Brinks+86",
        )
    print_minmax(f"Bubble {ylabel_for_print}", y_bub)
    if brinks is not None:
        print_minmax(f"Brinks {ylabel_for_print}", brinks[1])
    return bubble_handle, brinks_handle


def plot_panels():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    x_bub, bubble_div_nhi, bubble_div_kinetic, bubble_div_time = load_bubbles()
    brinks = load_brinks()
    x_turb, turb_nhi_lo, turb_nhi_hi, turb_h_lo, turb_h_hi, turb_i_lo, turb_i_hi = load_turbulence_band()
    brinks_nhi = None if brinks is None else (brinks[0], brinks[1], brinks[2])
    brinks_h = None if brinks is None else (brinks[0], brinks[2], brinks[1])
    brinks_i = None if brinks is None else (brinks[0], brinks[3], brinks[1])

    turb_nhi_lo, turb_nhi_hi = ordered_band(turb_nhi_lo, turb_nhi_hi)
    turb_h_lo, turb_h_hi = ordered_band(turb_h_lo, turb_h_hi)
    turb_i_lo, turb_i_hi = ordered_band(turb_i_lo, turb_i_hi)

    print_minmax("Turbulence panel density low", turb_nhi_lo)
    print_minmax("Turbulence panel density high", turb_nhi_hi)
    print_minmax("Turbulence panel h low", turb_h_lo)
    print_minmax("Turbulence panel h high", turb_h_hi)
    print_minmax("Turbulence panel i low", turb_i_lo)
    print_minmax("Turbulence panel i high", turb_i_hi)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_common(
        ax,
        "a",
        r"$\dot{e}/n_{\rm HI}$ [$10^{-28}$ erg s$^{-1}$ H$^{-1}$]",
        (5.0e-1, 2.0e3),
    )
    turb_handle = ax.fill_between(
        x_turb,
        turb_nhi_lo,
        turb_nhi_hi,
        color=TURB_COLOR,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
        zorder=10.25,
        label=r"turbulence / $n_{\rm HI}$",
    )
    bubble_handle, brinks_handle = scatter_points(ax, x_bub, bubble_div_nhi, brinks_nhi, "div nHI")
    handles = [bubble_handle]
    labels = [r"bubbles / $n_{\rm HI}$"]
    if brinks_handle is not None:
        handles.append(brinks_handle)
        labels.append(r"Brinks+86 / $n_{\rm HI}$")
    handles.append(turb_handle)
    labels.append(r"turbulence / $n_{\rm HI}$")
    leg = ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="black")
    leg.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    fig.savefig(DEFAULT_OUTPUT_NHI)
    plt.close(fig)
    print(f"[PanelsHI] Wrote {DEFAULT_OUTPUT_NHI}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_common(
        ax,
        "b",
        r"$\dot{e}/v^2$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$ (km s$^{-1}$)$^{-2}$]",
        (1.0e-4, 2.0),
    )
    turb_handle = ax.fill_between(
        x_turb,
        turb_h_lo,
        turb_h_hi,
        color=TURB_COLOR,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
        zorder=10.25,
        label=r"turbulence / $3\sigma_{\rm nt}^2$",
    )
    bubble_handle, brinks_handle = scatter_points(ax, x_bub, bubble_div_kinetic, brinks_h, "div velocity-squared")
    handles = [bubble_handle]
    labels = [r"bubbles / $v_{\rm exp}^2$"]
    if brinks_handle is not None:
        handles.append(brinks_handle)
        labels.append(r"Brinks+86 / $v_{\rm exp}^2$")
    handles.append(turb_handle)
    labels.append(r"turbulence / $3\sigma_{\rm nt}^2$")
    leg = ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="black")
    leg.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    fig.savefig(DEFAULT_OUTPUT_V2)
    plt.close(fig)
    print(f"[PanelsHI] Wrote {DEFAULT_OUTPUT_V2}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_common(
        ax,
        "c",
        r"$\dot{e}\,t$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$ Myr]",
        (8.0e-1, 1.0e4),
    )
    turb_handle = ax.fill_between(
        x_turb,
        turb_i_lo,
        turb_i_hi,
        color=TURB_COLOR,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
        zorder=10.25,
        label=r"turbulence $\times t_{\rm diss}$",
    )
    bubble_handle, brinks_handle = scatter_points(ax, x_bub, bubble_div_time, brinks_i, "times time")
    handles = [bubble_handle]
    labels = [r"bubbles $\times t_{\rm exp}$"]
    if brinks_handle is not None:
        handles.append(brinks_handle)
        labels.append(r"Brinks+86 $\times t_{\rm exp}$")
    handles.append(turb_handle)
    labels.append(r"turbulence $\times t_{\rm diss}$")
    leg = ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="black")
    leg.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    fig.savefig(DEFAULT_OUTPUT_TIME)
    plt.close(fig)
    print(f"[PanelsHI] Wrote {DEFAULT_OUTPUT_TIME}")


if __name__ == "__main__":
    plot_panels()
