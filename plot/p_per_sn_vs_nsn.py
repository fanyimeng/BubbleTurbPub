#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scatter: p_perSN (recomputed with selected SN) vs selected N_SN (fixed red markers; no data-driven coloring; edges off)

Selection rule (NO hybrid table):
- Weaver SN from column 'SN_weaver' (in ../code/1110.tab, scaled version).
- Chevalier (1974) eq.(26):
      E0[1e50 erg] = 5.3e-7 * n^1.12 * v^1.40 * R^3.12
      N_SN_chev    = E0 / 10
- If SN_weaver < N_SWITCH -> use Chevalier; else use Weaver.

Markers:
- Weaver-used      : circle 'o'
- Chevalier-used   : triangle '^'
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -----------------------
# Paths
# -----------------------
INPUT_TAB   = "../code/1113.tab"   # must contain p_rad_cgs, SN_weaver, n_HI_ring_cm-3
OUTPUT_PDF  = "p_per_sn_vs_nsn.pdf"

# -----------------------
# Switch threshold (Weaver -> Chevalier)
# -----------------------
N_SWITCH = 0.0001   # if SN_weaver < N_SWITCH, use Chevalier for that point

# -----------------------
# Plot controls
# -----------------------
COLOR_BY    = None     # column for coloring (if exists)
CMAP        = "coolwarm"
CBAR_LABEL  = r"$n_{\rm HI}$ (cm$^{-3}$)"

# —— Point style ——
POINT_SIZE_WEAVER   = 65
POINT_SIZE_CHEV     = 65
POINT_FACE_ALPHA    = 0.70
USE_DATA_EDGECOLOR  = False
POINT_EDGE_COLOR    = (0, 0, 0, 1.0)
POINT_EDGE_ALPHA    = 0.0
POINT_EDGE_LW       = 0.0
POINT_ZORDER        = 6

# Match the "New Bubbles" red used across the plotting suite
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)
BUBBLE_LABEL = "New Bubbles"

# —— Per-point error bars ——
X_ERR_FRAC = 0.7
Y_ERR_FRAC = 0.6
ERR_USE_DATA_COLOR = False
ERR_ECOLOR = (0.9, 0.4, 0.4, 0.35)
ERR_ELW    = 0.6
ERR_CAPSIZE= 0.0
ERR_LINESTYLE = '-'

# Axis labels & legend texts
X_LABEL = r"Estimated $N_{\rm SN}$ per bubble"
Y_LABEL = r"$p_{\rm SN}$ [$M_\odot$ km s$^{-1}$]"
SHADE_LEGEND_COMBINED = r"Theory ($n_{\rm HI} = 0.03\sim0.3\,{\rm cm^{-3}}$)"

# Whether to draw lines
DRAW_THEORY_LINE_CENTRAL = False
DRAW_DATA_FIT            = False

# Conversion (cgs -> Msun km/s)
G_PER_Msun = 1.989e33
CM_PER_KM  = 1e5
CGS_TO_Msun_kms = 1.0 / (G_PER_Msun * CM_PER_KM)

# ---------- Density range (units of m_H) ----------
RHO_CENTRE_mH = 1.33
RHO_RANGE_mH  = (1.33*0.03, 1.33*0.3)

# ---------- MANY regime parameters ----------
P0_many            = 23546 * 100.0
P0_many_ERR_PLUS   = 1072  * 100.0
P0_many_ERR_MINUS  = 1073  * 100.0
ETA_Z_many         = 0.15
ETA_Z_many_ERR     = 0.01
ETA_RHO_many       = 0.14
ETA_RHO_many_ERR   = 0.01
ETA_N_many         = -0.07
ETA_N_many_ERR     = 0.02
N_SCALE_many       = 1000.0

# ---------- FEW regime parameters ----------
P0_few             = 4249  * 100.0
P0_few_ERR_PLUS    = 741   * 100.0
P0_few_ERR_MINUS   = 683   * 100.0
ETA_Z_few          = 0.05
ETA_Z_few_ERR_UP   = 0.05
ETA_Z_few_ERR_DN   = 0.06
ETA_RHO_few        = -0.06
ETA_RHO_few_ERR    = 0.03
ETA_N_few          = 2.20
ETA_N_few_ERR_UP   = 0.24
ETA_N_few_ERR_DN   = 0.23
N_SCALE_few        = 1.0

# ---------- Sigma (use MANY's) ----------
SIGMA_CENTRAL  = 6075.0 * 100.0
SIGMA_ERR_PLUS = 214.0  * 100.0
SIGMA_ERR_MINUS= 202.0  * 100.0
USE_SIGMA = True

# plotting style
THEORY_LINESTYLE    = (0, (6, 3))
THEORY_LINEWIDTH    = 1.2
THEORY_LINECOLOR    = "0.15"
FILL_COLOR_COMBINED = (0.25, 0.25, 0.25, 0.22)
N_THEORY_SAMPLES    = 400

# Data-fit line style
FIT_LINESTYLE = "-"
FIT_LINEWIDTH = 1.3
FIT_COLOR     = "k"

# -----------------------
# Helpers
# -----------------------
def _finite_positive(*arrs):
    masks = [np.isfinite(np.asarray(a, float)) for a in arrs]
    m = np.logical_and.reduce(masks)
    for a in arrs:
        m &= (np.asarray(a, float) > 0)
    return m

def powerlaw_fit(x, y):
    lx = np.log10(x)
    ly = np.log10(y)
    k, a = np.polyfit(lx, ly, 1)
    A = 10 ** a
    ly_pred = k * lx + a
    ss_res = np.sum((ly - ly_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return A, k, r2

def p_per_SN_model(N, P0_val, etaZ, etaRho, etaN, rho_over_mH, Z_over_Zsun=1.0, N_scale=1000.0):
    N = np.asarray(N, dtype=float)
    return P0_val * (Z_over_Zsun ** etaZ) * (rho_over_mH ** etaRho) * ((N / N_scale) ** etaN)

def envelope_from_bounds(
    N, *,
    P0, dP0_plus, dP0_minus,
    etaZ, d_etaZ_plus, d_etaZ_minus,
    etaRho, d_etaRho,
    etaN, d_etaN_plus, d_etaN_minus,
    rho_range, rho_centre,
    Z_over_Zsun, N_scale,
    use_sigma, sigma_central, sigma_plus, sigma_minus,
):
    N = np.asarray(N, dtype=float)
    y_c = p_per_SN_model(N, P0, etaZ, etaRho, etaN, rho_centre, Z_over_Zsun, N_scale)

    P0_set   = (P0 - dP0_minus, P0 + dP0_plus)
    etaZ_set = (etaZ - d_etaZ_minus, etaZ + d_etaZ_plus)
    etaR_set = (etaRho - d_etaRho,   etaRho + d_etaRho)
    etaN_set = (etaN - d_etaN_minus, etaN + d_etaN_plus)
    rho_set  = (rho_range[0], rho_range[1])

    y_min = np.full_like(N, np.inf, dtype=float)
    y_max = np.full_like(N, -np.inf, dtype=float)
    for P0i in P0_set:
        for eZ in etaZ_set:
            for eR in etaR_set:
                for eN in etaN_set:
                    for rho in rho_set:
                        y = p_per_SN_model(N, P0i, eZ, eR, eN, rho, Z_over_Zsun, N_scale)
                        y_min = np.minimum(y_min, y)
                        y_max = np.maximum(y_max, y)

    if use_sigma:
        s_low  = sigma_central - sigma_minus
        s_high = sigma_central + sigma_plus
        y_min = np.maximum(0.0, y_min - s_low)
        y_max = y_max + s_high

    return y_c, y_min, y_max

def hmean_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (a * b) / (a + b)

def _read_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_fwf(path, header=0, infer_nrows=int(1e6))

def _pull_col(df, prefer, *alts):
    names = (prefer,) + alts
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy(float)
    raise KeyError(f"None of columns {names} found in {INPUT_TAB}")

# Chevalier eq.(26)
CHEV_C = 5.3e-7
def SN_chevalier(n_cm3, v_kms, R_pc):
    E0_1e50 = CHEV_C * (n_cm3**1.12) * (v_kms**1.40) * (R_pc**3.12)
    return E0_1e50 / 10.0

# -----------------------
# Main
# -----------------------
def main():
    # ---- Read table ----
    df = _read_table(INPUT_TAB)

    # required cols
    if "p_rad_cgs" not in df.columns:
        raise KeyError(f"'p_rad_cgs' missing in {INPUT_TAB}")
    if "SN_weaver" not in df.columns:
        raise KeyError(f"'SN_weaver' missing in {INPUT_TAB}")
    if "n_HI_ring_cm-3" not in df.columns:
        raise KeyError(f"'n_HI_ring_cm-3' missing in {INPUT_TAB} for Chevalier")

    # R & v with flexible names
    R_pc  = _pull_col(df, "R_pc", "radius_pc")
    v_kms = _pull_col(df, "v_kms", "expansion_vel")

    # read values
    p_rad  = pd.to_numeric(df["p_rad_cgs"],      errors="coerce").to_numpy(float)
    n_cm3  = pd.to_numeric(df["n_HI_ring_cm-3"], errors="coerce").to_numpy(float)
    SN_w   = pd.to_numeric(df["SN_weaver"],      errors="coerce").to_numpy(float)
    SN_c   = SN_chevalier(n_cm3, v_kms, R_pc)

    # selection by threshold
    use_chev = (SN_w < float(N_SWITCH)) & np.isfinite(SN_w) & np.isfinite(SN_c)
    SN_sel   = np.where(use_chev, SN_c, SN_w)

    # compute p_perSN with selected divisor
    with np.errstate(divide="ignore", invalid="ignore"):
        ppsn_cgs = np.where((SN_sel > 0) & np.isfinite(SN_sel), p_rad / SN_sel, np.nan)

    # mask
    m_use = _finite_positive(SN_sel, ppsn_cgs)
    if not np.any(m_use):
        raise RuntimeError("No valid rows after filtering finite/positive N_SN and p_rad/N_SN.")

    # arrays for plotting
    N_all   = SN_sel[m_use]
    ppsn    = ppsn_cgs[m_use] * CGS_TO_Msun_kms

    # ---------------- Axis ranges with margins ----------------
    y_min, y_max = ppsn.min() / 3.0, ppsn.max() * 3.0
    x_min, x_max = N_all.min() * 0.8,  N_all.max() * 1.2

    # theory x-grid
    x_theo = np.logspace(np.log10(min(x_min, 1e-3)), np.log10(x_max*50), N_THEORY_SAMPLES)

    # ---- Theory envelopes (many & few) ----
    _, yL_many, yU_many = envelope_from_bounds(
        x_theo,
        P0=P0_many, dP0_plus=P0_many_ERR_PLUS, dP0_minus=P0_many_ERR_MINUS,
        etaZ=ETA_Z_many, d_etaZ_plus=ETA_Z_many_ERR, d_etaZ_minus=ETA_Z_many_ERR,
        etaRho=ETA_RHO_many, d_etaRho=ETA_RHO_many_ERR,
        etaN=ETA_N_many, d_etaN_plus=ETA_N_many_ERR, d_etaN_minus=ETA_N_many_ERR,
        rho_range=RHO_RANGE_mH, rho_centre=RHO_CENTRE_mH,
        Z_over_Zsun=1.0, N_scale=N_SCALE_many,
        use_sigma=USE_SIGMA, sigma_central=SIGMA_CENTRAL,
        sigma_plus=SIGMA_ERR_PLUS, sigma_minus=SIGMA_ERR_MINUS
    )

    _, yL_few, yU_few = envelope_from_bounds(
        x_theo,
        P0=P0_few, dP0_plus=P0_few_ERR_PLUS, dP0_minus=P0_few_ERR_MINUS,
        etaZ=ETA_Z_few, d_etaZ_plus=ETA_Z_few_ERR_UP, d_etaZ_minus=ETA_Z_few_ERR_DN,
        etaRho=ETA_RHO_few, d_etaRho=ETA_RHO_few_ERR,
        etaN=ETA_N_few, d_etaN_plus=ETA_N_few_ERR_UP, d_etaN_minus=ETA_N_few_ERR_DN,
        rho_range=RHO_RANGE_mH, rho_centre=RHO_CENTRE_mH,
        Z_over_Zsun=1.0, N_scale=N_SCALE_few,
        use_sigma=USE_SIGMA, sigma_central=SIGMA_CENTRAL,
        sigma_plus=SIGMA_ERR_PLUS, sigma_minus=SIGMA_ERR_MINUS
    )

    # combined band via harmonic-mean of bounds
    yL_comb = hmean_pair(yL_few, yL_many)
    yU_comb = hmean_pair(yU_few, yU_many)

    # include band into y-range + floors/ceilings
    y_min = min(y_min, float(np.nanmin(yL_comb)))
    y_max = max(y_max, float(np.nanmax(yU_comb)))
    y_min = max(y_min, 1e5)
    y_max = max(y_max, 8e6)

    # FINAL axis limits with margins
    x_lo_hard, x_hi_hard = 0.5, 5e3
    x_lo = x_lo_hard / 1.05
    x_hi = x_hi_hard * 1.05
    y_lo = y_min / 1.15
    y_hi = y_max * 1.05

    # ---- Colors (from data column) ----
    cmap_obj = None
    norm = None
    if (COLOR_BY is not None) and (COLOR_BY in df.columns):
        col_all   = pd.to_numeric(df[COLOR_BY], errors="coerce").to_numpy(float)
        color_vals= col_all[m_use]
        cmap_obj  = plt.get_cmap(CMAP)
        norm      = plt.Normalize(vmin=np.nanmin(color_vals), vmax=np.nanmax(color_vals))
        colors = plt.get_cmap(CMAP)(norm(color_vals))
        facecolors = colors.copy()
        facecolors[:, 3] = POINT_FACE_ALPHA
        if USE_DATA_EDGECOLOR:
            edgecolors = colors.copy()
            edgecolors[:, 3] = POINT_EDGE_ALPHA
        else:
            ec = np.r_[POINT_EDGE_COLOR[:3], POINT_EDGE_ALPHA]
            edgecolors = np.repeat(ec[None, :], len(N_all), axis=0)
    else:
        face_rgba = np.array([BUBBLE_BASE_FACE_COLOR[0], BUBBLE_BASE_FACE_COLOR[1], BUBBLE_BASE_FACE_COLOR[2], POINT_FACE_ALPHA], dtype=float)
        facecolors = np.repeat(face_rgba[None, :], N_all.size, axis=0)
        ec = np.r_[POINT_EDGE_COLOR[:3], POINT_EDGE_ALPHA]
        edgecolors = np.repeat(ec[None, :], N_all.size, axis=0)

    # split by marker type
    use_chev_plot = use_chev[m_use]
    weaver_mask   = ~use_chev_plot
    chev_mask     =  use_chev_plot

    # ---- Fit (console only) ----
    A, k, r2 = powerlaw_fit(N_all, ppsn)
    print("=== Power-law fit to DATA (log-log) ===")
    print("Equation : y = A * x^k")
    print(f"A = {A:.6e}")
    print(f"k = {k:.4f}")
    print(f"R^2 = {r2:.4f}")
    print(f"N = {N_all.size}")
    print(f"[Switch] N_SWITCH={N_SWITCH:g}; Chevalier used on {int(np.sum(chev_mask))}/{N_all.size} points.")

    x_fit = np.logspace(np.log10(N_all.min()), np.log10(N_all.max()), 256)
    y_fit = A * x_fit ** k

    # ---- Plot setup ----
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", length=3)

    # draw combined theory shade
    ax.fill_between(x_theo, yL_comb, yU_comb, color=FILL_COLOR_COMBINED, linewidth=0, zorder=1)

    # optional central theory line
    if DRAW_THEORY_LINE_CENTRAL:
        yC_comb = hmean_pair(
            p_per_SN_model(x_theo, P0_few,  0.05, -0.06,  2.20, RHO_CENTRE_mH, 1.0, N_SCALE_few),
            p_per_SN_model(x_theo, P0_many, 0.15,  0.14, -0.07, RHO_CENTRE_mH, 1.0, N_SCALE_many)
        )
        ax.plot(x_theo, yC_comb, ls=THEORY_LINESTYLE, lw=THEORY_LINEWIDTH, color=THEORY_LINECOLOR, zorder=2)

    # colorbar
    if (COLOR_BY is not None) and (COLOR_BY in df.columns) and (cmap_obj is not None) and (norm is not None):
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=35)
        cbar.set_label(CBAR_LABEL, fontsize=9)
        cbar.ax.tick_params(labelsize=8, direction="in", length=4)

    # scatter by marker
    if np.any(weaver_mask):
        ax.scatter(
            N_all[weaver_mask], ppsn[weaver_mask],
            s=POINT_SIZE_WEAVER,
            facecolors=facecolors[weaver_mask],
            edgecolors="none",
            linewidths=POINT_EDGE_LW,
            marker='o',
            zorder=POINT_ZORDER,
            label="_nolegend_",
        )
    if np.any(chev_mask):
        ax.scatter(
            N_all[chev_mask], ppsn[chev_mask],
            s=POINT_SIZE_CHEV,
            facecolors=facecolors[chev_mask],
            edgecolors="none",
            linewidths=POINT_EDGE_LW,
            marker='^',
            zorder=POINT_ZORDER,
            label="_nolegend_",
        )

    # per-point error bars
    if ERR_USE_DATA_COLOR and (COLOR_BY in df.columns):
        for xi, yi, col in zip(N_all, ppsn, base_colors):
            xe = X_ERR_FRAC * float(xi)
            ye = Y_ERR_FRAC * float(yi)
            ecolor = (float(col[0]), float(col[1]), float(col[2]), 0.8)
            ax.errorbar([xi], [yi],
                        xerr=[[xe], [xe]], yerr=[[ye], [ye]],
                        fmt='none',
                        ecolor=ecolor,
                        elinewidth=ERR_ELW,
                        capsize=ERR_CAPSIZE,
                        ls=ERR_LINESTYLE,
                        zorder=POINT_ZORDER-1)
    else:
        xe_all = X_ERR_FRAC * N_all
        ye_all = Y_ERR_FRAC * ppsn
        for xi, yi, xe, ye in zip(N_all, ppsn, xe_all, ye_all):
            ax.errorbar([xi], [yi],
                        xerr=[[xe], [xe]], yerr=[[ye], [ye]],
                        fmt='none',
                        ecolor=ERR_ECOLOR,
                        elinewidth=ERR_ELW,
                        capsize=ERR_CAPSIZE,
                        ls=ERR_LINESTYLE,
                        zorder=POINT_ZORDER-1)

    # optional data fit line
    if DRAW_DATA_FIT:
        ax.plot(x_fit, y_fit, FIT_LINESTYLE, lw=FIT_LINEWIDTH, color=FIT_COLOR, zorder=4)

    # axes & labels
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    # legend
    handles = [Patch(facecolor=FILL_COLOR_COMBINED, edgecolor="none", label=SHADE_LEGEND_COMBINED)]
    handles.append(Line2D([], [], marker='o', linestyle='None',
                          markersize=np.sqrt(POINT_SIZE_WEAVER),
                          markerfacecolor=BUBBLE_BASE_FACE_COLOR,
                          markeredgecolor="none",
                          markeredgewidth=0.0,
                          alpha=POINT_FACE_ALPHA,
                          label=BUBBLE_LABEL))
    leg = ax.legend(handles=handles, loc="best", fontsize=9, frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    print(f"Saved: {OUTPUT_PDF}")
    plt.close()


if __name__ == "__main__":
    main()
