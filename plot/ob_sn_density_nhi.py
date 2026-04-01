
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
obsne_norm — Volume densities for OB (binned) + per-bubble SNe volume density as scatter
---------------------------------------------------------------------------------------
This version (v3+style+count+nHI-color, with tweaks):
  * 去掉网格；ticks 向内且四边显示。
  * x 轴标签统一为：Distance to M31's center: R [kpc]
  * 气泡（SNe scatter）默认更大。
  * 所有 bubble 数据从 ../code/1113.tab 读取。
  * 气泡散点在 y 轴仍为 SNe 体积密度，使用固定红色（不按数据染色），边框关闭。
  * Kang+2009 只画灰色阴影，不再画线。
  * bubble alpha = 0.7。
  * 统计并打印所选年龄范围内的 SF region 数目（段1/段2/并集）。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Optional, Tuple, List

# -------- constants --------
pc2cm = 3.08567758128e18
mH    = 1.6735575e-24
MU    = 1.4
COSI  = float(np.cos(np.deg2rad(77.0)))
ESN   = 1.0e51
RNG   = np.random.default_rng(12345)

BUBBLE_BASE_FACE_COLOR = (0.9, 0.4, 0.4)

# -------- legend helpers --------
def format_scale_tex(x: float, small: float = 1e-2, large: float = 1e3) -> str:
    if x == 1:
        return ""
    if x == 0 or not np.isfinite(x):
        return "0"
    ax = abs(x)
    if (ax < small) or (ax >= large):
        exp = int(np.floor(np.log10(ax)))
        mant = x / (10.0**exp)
        return rf"${mant:.2g}\times 10^{{{exp:d}}}$"
    else:
        return f"{x:g}"

def build_age_text(age_min, age_max) -> str:
    if age_min is None and age_max is None:
        return "OB (All Ages)"
    lo = "-∞" if age_min is None else f"{int(age_min)}"
    hi = "+∞" if age_max is None else f"{int(age_max)}"
    return f"OB ({lo}–{hi} Myr)"

def build_legend_text(scale: float, series_text: str, small=1e-2, large=1e3) -> str:
    return f"{format_scale_tex(scale, small, large)} {series_text}"

# -------- Kang+2009 reader --------
def read_kang2009_fwf(path: str) -> pd.DataFrame:
    colspecs = [
        (1-1,4),(6-1,14),(16-1,24),
        (26-1,31),(33-1,37),(39-1,44),(46-1,50),
        (52-1,55),(57-1,63),
        (65-1,69),(71-1,75),(77-1,81),
        (83-1,90),(92-1,99),(101-1,108),
        (110-1,114),(116-1,120),(122-1,126),
        (128-1,135),(137-1,144),(146-1,153)
    ]
    names = [
        "ID","RAdeg","DEdeg","FUVmag","e_FUVmag","NUVmag","e_NUVmag","EBV",
        "Area_arcsec2","Age_02","b_Age_02","B_Age_02","M_02","b_M_02","B_M_02",
        "Age_05","b_Age_05","B_Age_05","M_05","b_M_05","B_M_05"
    ]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None, dtype=str)
    for c in names:
        if c == "ID":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Age_02","b_Age_02","B_Age_02","Age_05","b_Age_05","B_Age_05"]:
        df.loc[np.isclose(df[c], -99.0, equal_nan=False), c] = np.nan
    for c in ["M_02","b_M_02","B_M_02","M_05","b_M_05","B_M_05"]:
        df.loc[np.isclose(df[c], -9.9e+01, equal_nan=False), c] = np.nan
    return df

# -------- FITS helpers --------
def world_to_pixel_rfits(r_fits_path: str, ra_deg, dec_deg):
    hdr = fits.getheader(r_fits_path)
    w = WCS(hdr)
    sky = SkyCoord(ra=np.asarray(ra_deg)*u.deg, dec=np.asarray(dec_deg)*u.deg)
    x, y = w.world_to_pixel(sky)
    return x, y

def get_fits_at_pix(fits_file: str, x_pix, y_pix) -> np.ndarray:
    data = fits.getdata(fits_file)
    x_pix = np.asarray(x_pix).astype(int)
    y_pix = np.asarray(y_pix).astype(int)
    vals = np.full(len(x_pix), np.nan, dtype=float)
    ok = (x_pix >= 0) & (x_pix < data.shape[1]) & (y_pix >= 0) & (y_pix < data.shape[0])
    vals[ok] = data[y_pix[ok], x_pix[ok]]
    return vals

# -------- geometry --------
def scale_height_pc(R_kpc: np.ndarray) -> np.ndarray:
    return 182.0 + 16.0 * np.asarray(R_kpc, float)

def build_bin_edges(R, n_bins=20, r_min=None, r_max=None):
    R = np.asarray(R, float)
    if r_min is None:
        r_min = np.nanmin(R)
    if r_max is None:
        r_max = np.nanmax(R)
    return np.linspace(r_min, r_max, n_bins+1)

def annulus_geo_from_edges(edges: np.ndarray):
    Rmid = 0.5*(edges[:-1] + edges[1:])
    dR   = edges[1:] - edges[:-1]
    area = 2*np.pi*Rmid*dR
    hpc  = scale_height_pc(Rmid)
    vol  = area * (2*hpc/1000.0)  # kpc^3
    return Rmid, area, hpc, vol

# -------- binning --------
def bin_sum_count_lists(R_kpc, weights, edges):
    nb = len(edges)-1
    sum_w  = np.zeros(nb)
    countk = np.zeros(nb, int)
    lists  = []
    for i in range(nb):
        sel = (R_kpc >= edges[i]) & (R_kpc < edges[i+1]) & np.isfinite(weights)
        w_i = weights[sel]
        sum_w[i]  = np.nansum(w_i)
        countk[i] = w_i.size
        lists.append(w_i.copy())
    return sum_w, countk, lists

# -------- uncertainty bands --------
def band_fractional(y, frac):
    dy = frac * y
    return np.clip(y-dy, 0, np.inf), y+dy

def band_poisson(y, k):
    k_safe = np.maximum(k.astype(float), 1.0)
    dy = y / np.sqrt(k_safe)
    return np.clip(y-dy, 0, np.inf), y+dy

def band_bootstrap(weight_lists, normalizer, B=2000, ci=0.68, rng=None):
    if rng is None:
        rng = RNG
    alpha = (1-ci)/2
    plo, phi = 100*alpha, 100*(1-alpha)
    low, high = np.full(len(weight_lists), np.nan), np.full(len(weight_lists), np.nan)
    for i,w in enumerate(weight_lists):
        n = len(w)
        if n==0:
            continue
        idx = rng.integers(0,n,size=(B,n))
        samples = w[idx]
        dens = np.nansum(samples,axis=1)/normalizer[i]
        low[i]  = np.nanpercentile(dens, plo)
        high[i] = np.nanpercentile(dens, phi)
    return low, high

# -------- SN per bubble --------
def compute_SN_per_bubble(r_pc, v_kms, NHI, h_pc):
    r_cm = r_pc * pc2cm
    v = v_kms * 1e5
    nHI = NHI / (2*h_pc*pc2cm/COSI)
    rho = MU*mH*nHI
    Eb = (4/3)*np.pi*(r_cm**3)*0.5*rho*(v**2)
    return Eb/ESN*(11/3)

# -------- plotting --------
def plot_one_axis_OB_and_SNe(
    R_bins, y_ob1, y_ob2, band_ob1, band_ob2,
    R_bub, rho_bub, xerr_bub=None, yerr_bub=None,
    c_bub=None,  # kept for backward compatibility; ignored (no data-driven coloring)
    *,
    DRAW_OB2=True,
    scale_ob1, scale_ob2, scale_sne,
    label_ob1, label_ob2, label_sne,
    out_pdf="ob_sn_density_nhi.pdf",
    figsize=(4.5,4),
    xlim=None, ylim=None,
    y_label="$\\rho_{*}$ [$10^{3}\\,M_{\\odot}\\,{\\rm kpc^{-3}}$]",
    x_label="Distance to M31's center: $R$ [kpc]",
    # OB bands: 灰色 shade
    color_ob1=(0.5, 0.5, 0.5),
    color_ob2=(0.5, 0.5, 0.5),
    lw_ob1=0, lw_ob2=0,    # 不画线
    band_ob1_alpha=0.45, band_ob2_alpha=0.38,
    # SNe scatter
    sne_marker_size=32,
    sne_alpha=0.70,        # bubble alpha = 0.7
    sne_edgecolor="none",
    sne_edgewidth=0.0,
    sne_zorder=2,
    # SNe errorbar
    sne_errorbar=True,
    sne_err_color=(0.2,0.2,0.2,0.0),
    sne_err_elinewidth=0.6,
    sne_err_capsize=0.0,
    sne_err_linestyle='-',
    sne_err_zorder=1,
):
    plt.rcParams['figure.dpi']=75
    plt.rcParams['savefig.dpi']=75
    plt.rcParams['axes.labelsize']=10
    plt.rcParams['xtick.labelsize']=10
    plt.rcParams['ytick.labelsize']=10

    fig, ax = plt.subplots(figsize=figsize)

    # ---------- OB SHADES ONLY, NO LINES ----------
    if band_ob1 is not None:
        lo, hi = band_ob1
        ax.fill_between(
            R_bins,
            lo*scale_ob1, hi*scale_ob1,
            color=color_ob1,
            alpha=band_ob1_alpha,
            lw=0,
            zorder=1,
            label=label_ob1  # 把标签挂在第一个 band 上
        )

    if DRAW_OB2 and (band_ob2 is not None):
        lo, hi = band_ob2
        ax.fill_between(
            R_bins,
            lo*scale_ob2, hi*scale_ob2,
            color=color_ob2,
            alpha=band_ob2_alpha,
            lw=0,
            zorder=1,
        )

    # ---------- bubble scatter ----------
    y_scat = rho_bub * scale_sne

    ax.scatter(
        R_bub,
        y_scat,
        s=sne_marker_size,
        facecolor=(BUBBLE_BASE_FACE_COLOR[0], BUBBLE_BASE_FACE_COLOR[1], BUBBLE_BASE_FACE_COLOR[2], sne_alpha),
        edgecolor=sne_edgecolor,
        linewidths=sne_edgewidth,
        zorder=sne_zorder,
        label=label_sne,
    )

    # ---------- errorbars ----------
    if sne_errorbar and (xerr_bub is not None or yerr_bub is not None):
        ax.errorbar(
            R_bub, y_scat,
            xerr=xerr_bub,
            yerr=None if yerr_bub is None else yerr_bub*scale_sne,
            fmt="none",
            ecolor=sne_err_color,
            elinewidth=sne_err_elinewidth,
            capsize=sne_err_capsize,
            ls=sne_err_linestyle,
            zorder=sne_err_zorder,
        )

    # ---------- axes ----------
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(direction="in", which="both", length=5,
                   top=True, bottom=True, left=True, right=True)

    leg = ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)

# -------- main --------
def main(
    kang_path="../code/kang_09_table2.dat",
    r_fits_path="../data/r-m31-jcomb_modHeader.fits",
    bubble_tab_path="../code/1113.tab",
    mom0_fits_path="../data/jcomb_vcube_scaled2_mom0.fits",
    alpha_fits_path="../data/alpha-m31-jcomb_modHeader.fits",
    AGE_COL="Age_02",
    AGE1_MIN=10.0, AGE1_MAX=50.0,
    AGE2_MIN=None, AGE2_MAX=None,
    DRAW_OB2=True,
    N_BINS=12, R_MIN=5.0, R_MAX=28.0,
    SCALE_OB1=1.0/5e1,
    SCALE_OB2=1.0/5e3,
    SCALE_SNE=1.0,
    UNCERT_OB1="poisson",
    UNCERT_OB2="poisson",
    BAND_OB1=0.3, BAND_OB2=0.3,
    BOOT_B=2000, BOOT_CI=0.68,
    XLIM=(5,28), YLIM=(-3,35),
    out_pdf="ob_sn_density_nhi.pdf",
    # SNe scatter style
    SNE_MARKER_SIZE=32,
    SNE_ALPHA=0.70,
    SNE_EDGECOLOR="none",
    SNE_EDGEWIDTH=0.0,
    SNE_ZORDER=2,
    SNE_ERRORBAR=True,
    SNE_ERR_COLOR=(0.2,0.2,0.2,0.0),
    SNE_ERR_ELINEWIDTH=0.6,
    SNE_ERR_CAPSIZE=0.0,
    SNE_ERR_LINESTYLE='-',
    SNE_ERR_ZORDER=1,
    # y 误差占位
    SNE_YERR_FRAC=0.50,
    # (kept) bubble n_HI column name (no longer used for coloring)
    BUBBLE_NHI_COL="n_HI_ring_cm-3",
):
    # Kang+2009
    df = read_kang2009_fwf(kang_path)
    xpix, ypix = world_to_pixel_rfits(r_fits_path, df["RAdeg"], df["DEdeg"])
    R_pc = get_fits_at_pix(r_fits_path, xpix, ypix)
    R_kpc = R_pc/1000.0
    M02 = df["M_02"].to_numpy(float)
    ages = df[AGE_COL].to_numpy(float)

    def in_range(a, lo, hi):
        m = np.isfinite(a)
        if lo is not None:
            m &= (a>=lo)
        if hi is not None:
            m &= (a<=hi)
        return m

    mask1 = in_range(ages, AGE1_MIN, AGE1_MAX) & np.isfinite(M02)
    mask2 = in_range(ages, AGE2_MIN, AGE2_MAX) & np.isfinite(M02)

    # 统计所选年龄范围的 SF region 数目
    n_sf1 = int(np.count_nonzero(mask1))
    print(f"[K09] SF regions in {AGE_COL} ∈ [{AGE1_MIN}, {AGE1_MAX}] Myr : {n_sf1}")
    if (AGE2_MIN is not None) or (AGE2_MAX is not None):
        n_sf2 = int(np.count_nonzero(mask2))
        lo2 = "-∞" if AGE2_MIN is None else AGE2_MIN
        hi2 = "+∞" if AGE2_MAX is None else AGE2_MAX
        print(f"[K09] SF regions in {AGE_COL} ∈ [{lo2}, {hi2}] Myr : {n_sf2}")
    n_union = int(np.count_nonzero(mask1 | mask2))
    print(f"[K09] SF regions in the UNION of selected ranges : {n_union}")

    edges = build_bin_edges(R_kpc, n_bins=N_BINS, r_min=R_MIN, r_max=R_MAX)
    Rmid, area, h_pc_bins, volume = annulus_geo_from_edges(edges)

    # OB densities (volume)
    sum1,k1,l1 = bin_sum_count_lists(R_kpc[mask1], M02[mask1], edges)
    rho1 = sum1/volume
    sum2,k2,l2 = bin_sum_count_lists(R_kpc[mask2], M02[mask2], edges)
    rho2 = sum2/volume

    # uncertainty
    def band(mode, y, frac, k, lists):
        if mode=="fractional":
            return band_fractional(y, frac)
        if mode=="poisson":
            return band_poisson(y, k)
        if mode=="bootstrap":
            return band_bootstrap(lists, volume, B=BOOT_B, ci=BOOT_CI)
        return None

    b1 = band(UNCERT_OB1, rho1, BAND_OB1, k1, l1)
    b2 = band(UNCERT_OB2, rho2, BAND_OB2, k2, l2)

    # SNe scatter data & error (per-bubble volume density)
    bub = pd.read_fwf(bubble_tab_path, header=0, infer_nrows=int(1e6))
    bx,by = bub["ra_pix"], bub["dec_pix"]
    R_pc_bub  = get_fits_at_pix(r_fits_path, bx, by)
    R_kpc_bub = R_pc_bub/1000.0
    mom0 = get_fits_at_pix(mom0_fits_path, bx, by)
    NHI = mom0*1.222e6/1.42/3600.*1.823e18
    h_pc = 182.0 + 16.0*R_kpc_bub
    vexp = bub["expansion_vel"].to_numpy(float) + 1.0
    r_pc = bub["radius_pc"].to_numpy(float)
    Nsn = compute_SN_per_bubble(r_pc, vexp, NHI, h_pc)
    r_kpc = r_pc/1000.0
    Vbub = (4/3)*np.pi*(r_kpc**3)
    rho_bub = np.divide(Nsn, Vbub, out=np.full_like(Nsn, np.nan), where=Vbub>0)

    # bubble x 误差（几何）
    angles = get_fits_at_pix(alpha_fits_path, bx, by) - 90.0
    dh = h_pc * np.tan(np.deg2rad(77.0)) * np.cos(np.deg2rad(angles))
    dxk = np.sqrt(dh**2 + r_pc**2) / 1e3  # kpc
    xerr = dxk

    # y 误差（占位比例）
    yerr = np.abs(SNE_YERR_FRAC * rho_bub)

    c_bub = None

    # legends
    leg1 = "K09 (10–50 Myr)"
    leg2 = build_legend_text(SCALE_OB2, build_age_text(AGE2_MIN, AGE2_MAX))
    legs = "Bubbles (100 $M_{\\odot}$ per SN)"

    plot_one_axis_OB_and_SNe(
        R_bins=Rmid, y_ob1=rho1, y_ob2=rho2, band_ob1=b1, band_ob2=b2,
        R_bub=R_kpc_bub, rho_bub=rho_bub, xerr_bub=xerr, yerr_bub=yerr,
        c_bub=c_bub,
        DRAW_OB2=DRAW_OB2,
        scale_ob1=SCALE_OB1, scale_ob2=SCALE_OB2, scale_sne=SCALE_SNE,
        label_ob1=leg1, label_ob2=leg2, label_sne=legs,
        out_pdf=out_pdf, xlim=XLIM, ylim=YLIM,
        sne_marker_size=SNE_MARKER_SIZE,
        sne_alpha=SNE_ALPHA,
        sne_edgecolor=SNE_EDGECOLOR,
        sne_edgewidth=SNE_EDGEWIDTH,
        sne_zorder=SNE_ZORDER,
        sne_errorbar=SNE_ERRORBAR,
        sne_err_color=SNE_ERR_COLOR,
        sne_err_elinewidth=SNE_ERR_ELINEWIDTH,
        sne_err_capsize=SNE_ERR_CAPSIZE,
        sne_err_linestyle=SNE_ERR_LINESTYLE,
        sne_err_zorder=SNE_ERR_ZORDER,
    )

if __name__ == "__main__":
    # 示例调用：bubble 来自 ../code/1113.tab（固定红色，不按数据染色）
    main(
        SCALE_OB1=1/1e3/1,
        SCALE_OB2=1.0/5e3,
        SCALE_SNE=100/1e3,
        DRAW_OB2=False,

        SNE_MARKER_SIZE=32,
        SNE_ALPHA=0.70,
        SNE_EDGECOLOR="none",
        SNE_EDGEWIDTH=0.0,

        SNE_ERRORBAR=True,
        SNE_ERR_COLOR=(0.2,0.2,0.2,0.0),
        SNE_ERR_ELINEWIDTH=0.6,
        SNE_ERR_CAPSIZE=0.0,
        SNE_ERR_LINESTYLE='-',
        SNE_ERR_ZORDER=1,
        SNE_YERR_FRAC=0.50,

        bubble_tab_path="../code/1113.tab",
        BUBBLE_NHI_COL="n_HI_ring_cm-3",
        out_pdf="ob_sn_density_nhi.pdf",
    )
