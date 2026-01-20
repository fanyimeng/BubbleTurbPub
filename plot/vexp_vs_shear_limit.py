#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSF shear-limit projection for M31 bubbles — styled scatter plot
(Fixed red bubble markers; no data-driven coloring.)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

# =========================
# User-configurable paths
# =========================
BUBBLE_TABLE_PATH = Path("../code/1113.tab")
ALPHA_FITS_PATH   = Path("../data/alpha-m31-jcomb_modHeader.fits")
MOM0_FITS_PATH    = Path("../data/jcomb_vcube_scaled2_mom0.fits")
RMAP_FITS_PATH    = Path("../data/r-m31-jcomb_modHeader.fits")

# === Optional exports (toggle to emit CSV/TXT summaries) ===
EXPORT_TABLES = False  # set True to write CSV + TXT alongside the PDF
OUT_CSV_PATH      = Path("./vexp_vs_shear_limit_projection_limits.csv")
OUT_SUMMARY_PATH  = Path("./vexp_vs_shear_limit_summary.txt")

# PDF always written (basename = script stem)
OUT_PDF_PATH      = Path("./vexp_vs_shear_limit.pdf")

# =========================
# Constants & geometry
# =========================
V_FLAT_KMS = 220.0
INC_DEG    = 77.0
INC_RAD    = np.deg2rad(INC_DEG)

pc2cm = 3e18
SN_BOUNDS = np.array([0, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200], dtype=float)

BUBBLE_MARKER_SIZE = 65
BUBBLE_MARKER = "o"
BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)
BUBBLE_FACE_ALPHA = 0.7
BUBBLE_EDGE_LW = 0.0
BUBBLE_LABEL = "New Bubbles"

ONE_TO_ONE_COLOR = (0.4, 0.4, 0.8, 0.8)
ONE_TO_ONE_LW = 1.2
ONE_TO_ONE_LS = "--"
ONE_TO_ONE_LABEL = "1:1"

LEGEND_LOC = "lower right"
LEGEND_FONTSIZE = 8
LEGEND_FRAME = True
LEGEND_FRAMEALPHA = 0.9
LEGEND_EDGE_COLOR = "black"
LEGEND_EDGE_LW = 0.5

# =========================
# Helpers
# =========================
def get_fits_at_pix(fits_file, x_pix, y_pix):
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan, dtype=float)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals


def read_bubble_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    except Exception:
        return pd.read_csv(path)

# =========================
# Load inputs
# =========================
df = read_bubble_table(BUBBLE_TABLE_PATH)

required_base = ["id", "ra_pix", "dec_pix", "radius_pc", "r_kpc", "expansion_vel"]
missing = [c for c in required_base if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in bubble table: {missing}")

x_pix = df["ra_pix"].astype(int).values
y_pix = df["dec_pix"].astype(int).values

# α: in-plane azimuth angle map (deg)
alpha_deg = get_fits_at_pix(ALPHA_FITS_PATH, x_pix, y_pix)
alpha_rad = np.deg2rad(alpha_deg)

# Bubble & radius geometry
r_bub_kpc = df["radius_pc"].astype(float).values / 1000.0
R_kpc     = df["r_kpc"].astype(float).values
v_obs     = df["expansion_vel"].astype(float).values

# Projected shear limit
with np.errstate(divide="ignore", invalid="ignore"):
    proj_fac   = np.sin(INC_RAD) * np.abs(np.cos(alpha_rad))
    v_lim_proj = (V_FLAT_KMS / R_kpc) * r_bub_kpc * proj_fac

bad = ~np.isfinite(v_lim_proj)
v_lim_proj[bad] = np.nan

viol_mask = (v_obs > v_lim_proj) & np.isfinite(v_lim_proj)
n_all     = len(v_obs)
n_valid   = np.count_nonzero(np.isfinite(v_lim_proj))
n_viol    = np.count_nonzero(viol_mask)
pct       = (100.0 * n_viol / n_valid) if n_valid > 0 else np.nan

# n_HI from mom0 + scale height model
mom0_vals = get_fits_at_pix(MOM0_FITS_PATH, x_pix, y_pix)
x_pos_kpc = get_fits_at_pix(RMAP_FITS_PATH, x_pix, y_pix) / 1e3

N_HI = mom0_vals * 1.222e6 / 1.42 / 3600 * 1.823e18
h_pc = 182 + 16 * x_pos_kpc
cosi = np.cos(np.deg2rad(77.0))
n_HI = N_HI / (2 * h_pc * pc2cm / cosi)

# Total SN (仍保留，用于 CSV 和后续分析，但不再作染色)
total_SN = ((df['radius_pc'].values * pc2cm)**3) * np.pi * (4/3) \
           * 0.5 * n_HI * 1.4 * 1.67e-24 * (v_obs * 1e5)**2 / 1e51 * 11/3.

# 旧的 SN colormap 不再用于绘图，但变量保留以免其它地方引用出错
norm_sn  = BoundaryNorm(boundaries=SN_BOUNDS, ncolors=len(SN_BOUNDS)-1)
cmap_sn  = ListedColormap(plt.get_cmap('coolwarm')(np.linspace(0, 1, len(SN_BOUNDS)-1)))

# 输出表
out = df.copy()
out["alpha_deg"]     = alpha_deg
out["bubble_r_kpc"]  = r_bub_kpc
out["proj_factor"]   = proj_fac
out["v_lim_proj"]    = v_lim_proj
out["vexp_gt_vlim"]  = viol_mask
out["n_HI_cm-3"]     = n_HI
out["total_SN"]      = total_SN

# 文本 summary
summary_lines = [
    "NSF projected shear limit with flat RC (V=220 km/s)",
    "Formula: v_lim_proj = (V/R) * r_bub * sin(i) * |cos α|",
    f"i = {INC_DEG:.1f} deg",
    f"Input catalog: {BUBBLE_TABLE_PATH}",
    f"Angle FITS   : {ALPHA_FITS_PATH}",
    f"Valid bubbles (finite v_lim_proj): {n_valid} / {n_all}",
    f"Exceeding projected NSF limit    : {n_viol} / {n_valid} = {pct:.1f}%",
]

if EXPORT_TABLES:
    out.to_csv(OUT_CSV_PATH, index=False)
    OUT_SUMMARY_PATH.write_text("\n".join(summary_lines))

print("\n".join(summary_lines))

# =========================
# Plotting function
# =========================
def plot_vexp_vs_shear(
    xv,
    yv,
    figsize=(3, 3),
    out_path=OUT_PDF_PATH,
):
    """
    xv: v_lim_proj
    yv: v_obs
    """
    plt.rcParams['figure.dpi'] = 75.
    plt.rcParams['savefig.dpi'] = 75.

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # minor tick formatter
    for axis in [ax.yaxis]:
        axis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    # ticks
    ax.tick_params(axis='both', which='major', direction='in', size=10,
                   bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', size=6,
                   bottom=True, top=True, left=True, right=True)

    ax.set_xlabel(r"$v_{\rm lim,proj}$  (km s$^{-1}$)")
    ax.set_ylabel(r"$v_{\rm exp}$  (km s$^{-1}$)")
    ax.yaxis.labelpad = 2

    # 误差条：横向 25%，纵向 20%
    xerr = 0.25 * xv
    yerr = 0.20 * yv
    ax.errorbar(
        xv,
        yv,
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor="0.3",
        elinewidth=0.4,
        capsize=0,
        alpha=0.4,
        zorder=2,
    )

    # Scatter: fixed red markers (match plot/dot_e_vs_turb.py styling; no data-driven coloring)
    ax.scatter(
        xv,
        yv,
        s=BUBBLE_MARKER_SIZE,
        marker=BUBBLE_MARKER,
        facecolors=BUBBLE_BASE_FACE_COLOR,
        edgecolors="none",
        linewidths=BUBBLE_EDGE_LW,
        zorder=3,
        alpha=BUBBLE_FACE_ALPHA,
    )

    # 1:1 线 + 轴范围
    if xv.size and yv.size:
        mx = 1.05 * float(max(np.nanmax(xv), np.nanmax(yv)))
        ax.plot(
            [0, mx],
            [0, mx],
            color=ONE_TO_ONE_COLOR,
            lw=ONE_TO_ONE_LW,
            ls=ONE_TO_ONE_LS,
            zorder=1,
        )
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)

    ax.set_aspect("equal", adjustable="box")

    bubble_handle = Line2D(
        [0],
        [0],
        marker=BUBBLE_MARKER,
        linestyle="none",
        markersize=np.sqrt(BUBBLE_MARKER_SIZE),
        markerfacecolor=BUBBLE_BASE_FACE_COLOR,
        markeredgewidth=0.0,
        alpha=BUBBLE_FACE_ALPHA,
        label=BUBBLE_LABEL,
    )
    one_to_one_handle = Line2D(
        [0],
        [0],
        color=ONE_TO_ONE_COLOR,
        lw=ONE_TO_ONE_LW,
        ls=ONE_TO_ONE_LS,
        label=ONE_TO_ONE_LABEL,
    )
    leg = ax.legend(
        handles=[bubble_handle, one_to_one_handle],
        loc=LEGEND_LOC,
        fontsize=LEGEND_FONTSIZE,
        frameon=LEGEND_FRAME,
    )
    if LEGEND_FRAME:
        frame = leg.get_frame()
        frame.set_alpha(LEGEND_FRAMEALPHA)
        frame.set_edgecolor(LEGEND_EDGE_COLOR)
        frame.set_linewidth(LEGEND_EDGE_LW)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PDF: {out_path}")


# =========================
# 调用示例
# =========================
ok = np.isfinite(v_lim_proj) & np.isfinite(v_obs)
plot_vexp_vs_shear(
    v_lim_proj[ok],
    v_obs[ok],
    figsize=(4.5, 4),
    out_path=OUT_PDF_PATH,
)
