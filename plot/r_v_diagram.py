#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import FuncNorm, PowerNorm, Normalize
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import re
from pathlib import Path
DEBUG_PRINT_BRINKS = False

CONNECTION_COLOR = (0.6,0.6,0.6,0.8)

pc2cm = 3e18

def _read_rot_fits_to_arcmin_with_wcs(fits_path):
    """
    读取旋转后的 2D FITS，并返回 (data, extent_arcmin, header, wcs, cd1_am, cd2_am, nx, ny)。
    extent 以 arcmin 为单位：[xmin, xmax, ymin, ymax]，原点在图中心。
    """
    hdu = fits.open(fits_path)[0]
    data = hdu.data
    hdr = hdu.header
    wcs = WCS(hdr)

    nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
    cd1_am = float(abs(hdr.get("CDELT1", 0.0))) * 60.0  # arcmin/px
    cd2_am = float(abs(hdr.get("CDELT2", 0.0))) * 60.0  # arcmin/px

    x_center, y_center = nx/2.0, ny/2.0
    x_extent = (np.arange(nx) - x_center) * cd1_am
    y_extent = (np.arange(ny) - y_center) * cd2_am
    extent = [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]]
    return data, extent, hdr, wcs, cd1_am, cd2_am, nx, ny

def _rotate_xy(x_am, y_am, deg):
    """
    将以图心为原点的 arcmin 坐标 (x_am, y_am) 逆时针旋转 deg 度。
    支持标量或等长数组。
    """
    theta = np.radians(deg)
    ct, st = np.cos(theta), np.sin(theta)
    xr = x_am * ct - y_am * st
    yr = x_am * st + y_am * ct
    return xr, yr

def _shrink_row_axes_horiz(ax_list, frac):
    """
    将同一行的一组 axes 在水平方向整体按比例 frac(0~1) 居中收缩。
    对单个轴也适用（传 [ax] 即可）。
    """
    if not (0 < float(frac) <= 1.0):
        return
    x0 = min(ax.get_position().x0 for ax in ax_list)
    x1 = max(ax.get_position().x1 for ax in ax_list)
    row_width = x1 - x0
    new_width = row_width * float(frac)
    cx = (x0 + x1) * 0.5
    nx0 = cx - new_width * 0.5
    for ax in ax_list:
        p = ax.get_position()
        rx0 = (p.x0 - x0) / row_width
        rx1 = (p.x1 - x0) / row_width
        ax.set_position([nx0 + rx0 * new_width, p.y0,
                         (rx1 - rx0) * new_width, p.height])

def plot_five_panels_three_rows(
    # ===== 数据与表 =====
    fits_rotated_a='../data/jcomb_submed.mom0.-600-45_rotated.fits',
    table_path='../code/1113.tab',
    brinks_region_path="../data/b86_inpix_id.reg",
    brinks_region_show=True,
    brinks_region_marker="^",
    brinks_region_size=45,
    brinks_region_color=(0.6, 0.6, 0.9),
    brinks_region_edge_alpha=0.6,
    brinks_region_face_alpha=0.6,
    brinks_region_lw=0.0,
    brinks_region_alpha=0.7,
    brinks_region_zorder=6,
    brinks_table_path="../data/brinks+86/brinks86_combined.fwf",
    brinks_panel_e_show=True,
    brinks_panel_e_marker=None,
    brinks_panel_e_size=None,
    brinks_panel_e_color=None,
    brinks_panel_e_edge_alpha=None,
    brinks_panel_e_face_alpha=None,
    brinks_panel_e_lw=None,
    brinks_panel_e_alpha=None,
    brinks_panel_e_zorder=None,
    bubble_panel_a_marker="o",
    bubble_panel_a_size=20,
    bubble_panel_a_color=(0.9, 0.2, 0.2),
    bubble_panel_a_edge_alpha=0.0,
    bubble_panel_a_face_alpha=0.6,
    bubble_panel_a_lw=0.6,
    bubble_panel_a_edge_color=(0.9, 0.2, 0.2, 0),
    bubble_panel_a_alpha=0.6,
    bubble_panel_a_zorder=8,

    # ===== (a) 背景显示参数 =====
    a_norm_kind='power',             # 'power' 或 'asinh'
    a_power_gamma=0.6,
    a_vmin=None,
    a_vmax=None,
    a_xlim_arcmin=(-110, 110),
    a_ylim_arcmin=(-40, 40),
    direction_angle_a=-52,           # 竖直向上=0°，逆时针为正
    rot_delta_deg=52.0,              # 椭圆朝向 +52°
    rotate_bubble_pos_deg=52.0,      # 位置坐标也 +52°
    dist_mpc=0.761,
    scalebar_kpc=5.0,

    # ===== (b)(c)(d) 三幅子图（横向）=====
    fits_path_list_center=(
        '../data/b302_-94-88.fits',
        '../data/b302_-88-82.fits',
        '../data/b302_-82-76.fits'
    ),
    xlim_arcmin=(-5, 5),
    ylim_arcmin=(-5, 5),
    vmin_list_center=(0.01, 0.01, 0.01),
    vmax_list_center=(4.0, 4.0, 4.0),

    # 面板标题文本与字号（可调）
    title_a='(a)',
    title_b=r'(b) $(-94,-88)$',
    title_c=r'(c) $(-88,-82)$',
    title_d=r'(d) $(-82,-76)$',
    title_e='(e)',
    title_fs_a=11,
    title_fs_mid=9,
    title_fs_e=11,

    # 中行三个子图的 tick label 显示（可调）
    b_show_labels=True,    # (b) 显示坐标文字
    c_show_labels=False,   # (c) 不显示
    d_show_labels=False,   # (d) 不显示

    # 各面板刻度密度（可调）
    a_xtick_major=20.0,    a_ytick_major=10.0,
    mid_xtick_major=2.0,   mid_ytick_major=2.0,
    e_xtick_major=None,    e_ytick_major=None,  # 若为 None 则使用默认

    # ===== (e) r–vexp：按 n_ring 连续着色 =====
    color_vmin=None,       # 若为 None，则自动从 n_ring 计算
    color_vmax=None,       # 若为 None，则自动从 n_ring 计算
    color_cmap='coolwarm',
    color_by_nring=False,
    bubble_base_face_color=(0.9, 0.2, 0.2, 0.0),
    bubble_base_edge_color=(0.9, 0.2, 0.2, 0.0),
    bubble_base_edge_lw=0.0,
    r_curve_minmax=(90, 2000),
    n0=0.2,
    rsn_list=(10, 1, 0.1),
    T_list=(10, 20, 40),

    # (e) 点与误差棒样式（默认同 Energy budget figure）
    bubble_marker_size=75,
    bubble_marker_face_alpha=0.0,
    bubble_marker_edge_lw=0.0,
    bubble_marker_edge_alpha=0.0,
    bubble_marker_edge_color=(0.9, 0.2, 0.2, 0.0),
    bubble_marker_zorder=8,

    bubble_show_errorbar=True,
    bubble_errorbar_use_data_color=True,
    bubble_errorbar_data_alpha=0.8,
    bubble_errorbar_elinewidth=0.6,
    bubble_errorbar_capsize=0.0,
    bubble_errorbar_linestyle='-',
    bubble_errorbar_zorder=5,
    err_frac_x=0.25,  # x 相对不确定度（默认 25%）
    err_frac_y=0.25,  # y 相对不确定度（默认 25%）

    # ===== 高亮样式 =====
    highlight_id=None,

    # 可选：'ring'（默认双黑边黄环）、'star'（五角星）、'emph'（把 a 中该椭圆与 e 中该点改成亮黄+黑边）
    highlight_style='ring',
    hl_marker_size=220,                        # ring/star 的基准尺寸

    # —— Ring 三层样式：外黑边 + 黄环 + 内黑边 ——
    hl_ring_edgecolor=(1.0, 0.9, 0.2, 0.95),   # 中间黄环颜色
    hl_ring_linewidth=2.0,                     # 中间黄环线宽
    hl_ring_outer_edgecolor='k',               # 外黑边
    hl_ring_outer_linewidth=1.4,
    hl_ring_inner_edgecolor='k',               # 内黑边
    hl_ring_inner_linewidth=1.4,
    hl_ring_outer_scale=1.08,                  # 外黑边尺寸比例（相对基准环）
    hl_ring_inner_scale=0.86,                  # 内黑边尺寸比例

    # —— Star 参数（保留以便切换）——
    hl_star_facecolor=(1.0, 0.9, 0.2, 0.95),
    hl_star_edgecolor='k',
    hl_star_linewidth=0.9,

    # —— Emph 参数（统一后备值；若分开参数为 None，则回退到这组）——
    hl_emph_facecolor=(1.0, 0.9, 0.2, 0.95),   # 统一面色后备
    hl_emph_edgecolor='k',                     # 统一边色后备
    hl_emph_ellipse_lw=1.6,                    # 统一椭圆线宽后备
    hl_emph_point_size=140,                    # 统一点大小后备
    hl_emph_point_edge_lw=0.9,                 # 统一点边线宽后备

    # —— Emph (a) 椭圆专用（可与 (e) 分开）——
    hl_emph_a_facecolor=None,                  # 若 None 则用 hl_emph_facecolor
    hl_emph_a_edgecolor=None,                  # 若 None 则用 hl_emph_edgecolor
    hl_emph_a_ellipse_lw=None,                 # 若 None 则用 hl_emph_ellipse_lw

    # —— Emph (e) 散点专用（可与 (a) 分开）——
    hl_emph_e_facecolor=None,                  # 若 None 则用 hl_emph_facecolor
    hl_emph_e_edgecolor=None,                  # 若 None 则用 hl_emph_edgecolor
    hl_emph_e_point_size=None,                 # 若 None 则用 hl_emph_point_size
    hl_emph_e_point_edge_lw=None,              # 若 None 则用 hl_emph_point_edge_lw

    # ===== 连线：自动角点与手动覆盖/微调（默认微调为你提供的值）=====
    auto_corner_links=True,

    # 1) (a)->(c) 左上
    link_a2c_ul_enabled=True,
    link_a2c_ul_src_override=None,          # (x_am, y_am) in (a)
    link_a2c_ul_src_delta=(-0.0, -0.0),
    link_a2c_ul_tgt_override=None,          # (x_am, y_am) in (c)
    link_a2c_ul_tgt_delta=(+0.0, -0.0),
    link_a2c_ul_ls='--',
    link_a2c_ul_color=CONNECTION_COLOR,
    link_a2c_ul_lw=1.4,
    link_a2c_ul_alpha=1.0,

    # 2) (a)->(c) 右上
    link_a2c_ur_enabled=True,
    link_a2c_ur_src_override=None,
    link_a2c_ur_src_delta=(+0.0, -0.0),
    link_a2c_ur_tgt_override=None,
    link_a2c_ur_tgt_delta=(-0.0, -0.0),
    link_a2c_ur_ls='--',
    link_a2c_ur_color=CONNECTION_COLOR,
    link_a2c_ur_lw=1.4,
    link_a2c_ur_alpha=1.0,

    # 3) (e)->(c) 左下
    link_e2c_ll_enabled=True,
    link_e2c_ll_src_override=None,          # (r_pc, v_kms)
    link_e2c_ll_src_delta=(-0.0, +0.0),
    link_e2c_ll_tgt_override=None,          # (x_am, y_am) in (c)
    link_e2c_ll_tgt_delta=(-0.0, +0.0),
    link_e2c_ll_ls='--',
    link_e2c_ll_color=CONNECTION_COLOR,
    link_e2c_ll_lw=1.4,
    link_e2c_ll_alpha=1.0,

    # 4) (e)->(c) 右下
    link_e2c_lr_enabled=True,
    link_e2c_lr_src_override=None,
    link_e2c_lr_src_delta=(-0.0, +0.0),
    link_e2c_lr_tgt_override=None,
    link_e2c_lr_tgt_delta=(-0.0, +0.0),
    link_e2c_lr_ls='--',
    link_e2c_lr_color=CONNECTION_COLOR,
    link_e2c_lr_lw=1.4,
    link_e2c_lr_alpha=1.0,

    # ===== 布局（默认值按你的要求，并新增“三行宽度比例”可调）=====
    figsize=(5.0, 6.6),                  # 画布大小
    height_ratios=(1.0, 0.8, 3.0),       # 顶/中/底三行的相对高度
    hspace_top_mid=0.1,                  # 顶行与中行间距
    hspace_mid_bottom=0.1,               # 中行与底行间距
    mid_wspace=0.1,                      # 中行三图之间的水平间距
    mid_width_ratios=(1.0, 1.0, 1.0),    # 中行三图的相对宽度
    e_colorbar_size='4.5%',              # (e) 色条宽度（相对 ax_e 宽度）
    e_colorbar_pad=0.05,                 # (e) 色条与 panel e 之间的间距（相对 ax_e 宽度）
    width_frac_top=1.0,                  # 顶行整体可用宽度比例(0~1)
    width_frac_mid=1.0,                  # 中行整体可用宽度比例(0~1)
    width_frac_bottom=1.0,               # 底行整体可用宽度比例(0~1)

    # ===== 输出 =====
    output_name='r_v_diagram.pdf'
):
    """
    三行布局：
      顶行 (a)：旋转后的 FAST+VLA 图（arcmin 相对坐标）+ N/W 指向标 + 标尺/波束 + 椭圆（按 n_ring 连续着色）。
      中行 (b)(c)(d)：横向三幅（arcmin 相对坐标；b/c/d 的坐标文字是否显示、刻度密度都可调）。
      底行 (e)：r–vexp，气泡点与误差棒样式完全仿照 Energy budget figure，颜色连续映射到 n_ring（color_vmin–color_vmax，默认为数据范围）。
    """

    # ---------- 读表 ----------
    df = pd.read_fwf(table_path, header=0, infer_nrows=int(1e6))
    required_cols = ["radius_pc", "expansion_vel", "n_HI_ring_cm-3",
                     "ra_hms", "dec_dms", "maj_as", "min_as", "pa"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {table_path}: {missing}")

    r_pc   = df["radius_pc"].to_numpy(float)
    v_exp  = df["expansion_vel"].to_numpy(float)
    n_ring = df["n_HI_ring_cm-3"].to_numpy(float)

    has_id = ('id' in df.columns)

    # 自动确定 color_vmin / color_vmax（若为 None）
    if (color_vmin is None) or (color_vmax is None):
        finite_n = n_ring[np.isfinite(n_ring)]
        if finite_n.size > 0:
            data_min = float(np.nanmin(finite_n))
            data_max = float(np.nanmax(finite_n))
            if color_vmin is None:
                color_vmin = data_min
            if color_vmax is None:
                color_vmax = data_max
        else:
            # 若全是 NaN，退化为 [0,1]
            if color_vmin is None:
                color_vmin = 0.0
            if color_vmax is None:
                color_vmax = 1.0
    # 防止 vmax <= vmin 的极端情况
    if color_vmax <= color_vmin:
        eps = max(abs(color_vmin), 1.0) * 1e-3
        color_vmax = color_vmin + eps

    # ---------- (a) 顶行：背景 + WCS ----------
    data_a, extent_a, hdr_a, wcs_a, cd1_am, cd2_am, nx_a, ny_a = _read_rot_fits_to_arcmin_with_wcs(fits_rotated_a)

    # 背景强度归一
    if str(a_norm_kind).lower() == "power":
        norm_a = PowerNorm(gamma=a_power_gamma, vmin=a_vmin, vmax=a_vmax)
    else:
        norm_a = FuncNorm((np.arcsinh, np.sinh), vmin=a_vmin, vmax=a_vmax)

    # 连续色标：n_ring（供 (a) 椭圆与 (e) 氣泡复用）
    norm_n = Normalize(vmin=float(color_vmin), vmax=float(color_vmax))
    cmap_n = plt.get_cmap(color_cmap)

    # ---------- 中行 (b)(c)(d) 读入 ----------
    def _open_center_fits(path):
        freq_GHz = 1.4
        hdu = fits.open(path)[0]
        data = hdu.data
        hdr = hdu.header
        if 'BMAJ' not in hdr:
            raise ValueError(f"FITS 文件 {path} 缺少 BMAJ")
        beam_as = hdr['BMAJ'] * 3600.0  # arcsec
        # 粗略从亮温转换至列密度刻度（仅用于显示一致性）
        data = data * 1.222e6 / (freq_GHz**2 * beam_as**2) * 1.8e18 / 1e20
        nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
        cd1 = abs(hdr.get('CDELT1', 0.0)) * 60.0  # arcmin/px
        cd2 = abs(hdr.get('CDELT2', 0.0)) * 60.0
        xc, yc = nx/2.0, ny/2.0
        x_extent = (np.arange(nx) - xc) * cd1
        y_extent = (np.arange(ny) - yc) * cd2
        extent = [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]]
        return data, extent

    data_list_center, extent_list_center = [], []
    for p in fits_path_list_center:
        d, ext = _open_center_fits(p)
        data_list_center.append(d)
        extent_list_center.append(ext)

    # ---------- (e) r–vexp 的参考曲线 ----------
    r_vals = np.logspace(np.log10(r_curve_minmax[0]), np.log10(r_curve_minmax[1]), 200)

    # ---------- 画布与网格：三行 ----------
    plt.rcParams.update({
        "figure.dpi": 90,
        "savefig.dpi": 90,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=height_ratios)

    # ========== 顶行 (a) ==========
    ax_a = fig.add_subplot(gs[0, 0])
    fig.subplots_adjust(hspace=0.0)
    ax_a.imshow(data_a, origin="lower", cmap="gray", norm=norm_a, extent=extent_a)
    ax_a.set_xlim(a_xlim_arcmin)
    ax_a.set_ylim(a_ylim_arcmin)
    ax_a.tick_params(axis='both', color='white', direction='in')

    # a 面板刻度密度与标签显示
    if a_xtick_major is not None:
        ax_a.xaxis.set_major_locator(MultipleLocator(a_xtick_major))
    if a_ytick_major is not None:
        ax_a.yaxis.set_major_locator(MultipleLocator(a_ytick_major))
    ax_a.tick_params(labelbottom=True, labelleft=True, labelcolor='black')
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}'"))
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}'"))
    ax_a.text(0.02, 0.97, title_a, transform=ax_a.transAxes, fontsize=title_fs_a,
              color='white', va='top')

    # (a) 标尺（arcmin 单位）
    arcsec_per_rad = 206265.0
    theta_as = scalebar_kpc * 1e3 / (dist_mpc * 1e6) * arcsec_per_rad
    scalebar_arcmin = theta_as / 60.0
    x_min, x_max = a_xlim_arcmin
    y_min, y_max = a_ylim_arcmin
    x0 = x_min + 0.90 * (x_max - x_min)
    y0 = y_min + 0.10 * (y_max - y_min)
    ax_a.plot([x0 - scalebar_arcmin/2.0, x0 + scalebar_arcmin/2.0], [y0, y0],
              color='white', lw=1.0)
    ax_a.text(x0, y0 + 8.1, f"{scalebar_kpc:.0f} kpc", ha='center', va='top',
              fontsize=8, color='white')

    # (a) N/W 指向标
    arrow_base_x = x_min + 0.11 * (x_max - x_min)
    arrow_base_y = y_min + 0.08 * (y_max - y_min)
    arrow_len = 0.04 * (x_max - x_min)
    theta_dir = np.radians(direction_angle_a)
    dx_N = arrow_len * np.sin(theta_dir); dy_N = arrow_len * np.cos(theta_dir)
    ax_a.arrow(arrow_base_x, arrow_base_y, dx_N, dy_N,
               head_width=1, head_length=1, fc='white', ec='white', linewidth=0.7)
    ax_a.text(arrow_base_x + dx_N * 1.70, arrow_base_y + dy_N * 1.70, 'N',
              color='white', fontsize=8, ha='center', va='center')
    theta_W = theta_dir + np.pi / 2.0
    dx_W = arrow_len * np.sin(theta_W); dy_W = np.cos(theta_W) * arrow_len
    ax_a.arrow(arrow_base_x, arrow_base_y, dx_W, dy_W,
               head_width=1, head_length=1, fc='white', ec='white', linewidth=0.7)
    ax_a.text(arrow_base_x + dx_W * 1.70, arrow_base_y + dy_W * 1.70, 'W',
              color='white', fontsize=8, ha='center', va='center')

    # (a) 椭圆（位置坐标额外旋转 rotate_bubble_pos_deg）
    ra = df["ra_hms"].astype(str).values
    dec = df["dec_dms"].astype(str).values
    coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    x_pix, y_pix = wcs_a.world_to_pixel(coords)
    x_am = (x_pix - nx_a/2.0) * cd1_am
    y_am = (y_pix - ny_a/2.0) * cd2_am
    x_am_rot, y_am_rot = _rotate_xy(x_am, y_am, rotate_bubble_pos_deg)

    maj_as = df["maj_as"].to_numpy(float)
    min_as = df["min_as"].to_numpy(float)
    pa_deg = df["pa"].to_numpy(float)

    # 取出 emph 的 (a)/(e) 专用样式（若为 None 则回退到统一后备）
    _a_face = hl_emph_a_facecolor if (hl_emph_a_facecolor is not None) else hl_emph_facecolor
    _a_edge = hl_emph_a_edgecolor if (hl_emph_a_edgecolor is not None) else hl_emph_edgecolor
    _a_lw   = hl_emph_a_ellipse_lw if (hl_emph_a_ellipse_lw is not None) else hl_emph_ellipse_lw

    _e_face = hl_emph_e_facecolor if (hl_emph_e_facecolor is not None) else hl_emph_facecolor
    _e_edge = hl_emph_e_edgecolor if (hl_emph_e_edgecolor is not None) else hl_emph_edgecolor
    _e_size = hl_emph_e_point_size if (hl_emph_e_point_size is not None) else hl_emph_point_size
    _e_lw   = hl_emph_e_point_edge_lw if (hl_emph_e_point_edge_lw is not None) else hl_emph_point_edge_lw

    # 如果是 emph，提前把 highlight id 读出（若有）
    hl_id_val = None
    if highlight_id is not None and has_id:
        hl_id_val = highlight_id

    for i in range(len(df)):
        angle_plot = float(pa_deg[i]) + 90.0 + float(rot_delta_deg)

        if (hl_id_val is not None) and (df['id'].iloc[i] == hl_id_val) and (str(highlight_style).lower() == 'emph'):
            # 强调：亮黄填充 + 黑边（用 (a) 专用样式）
            ax_a.add_patch(Ellipse((x_am_rot[i], y_am_rot[i]),
                                   width=(min_as[i]/60.0)*2.0,  # arcmin
                                   height=(maj_as[i]/60.0)*2.0, # arcmin
                                   angle=angle_plot,
                                   facecolor=_a_face,
                                   edgecolor=_a_edge,
                                   lw=_a_lw,
                                   zorder=max(bubble_panel_a_zorder, 7)))
        else:
            # 常规：可按 n_ring 连续着色，也可单色
            if color_by_nring and np.isfinite(n_ring[i]):
                rgba = plt.get_cmap(color_cmap)(norm_n(n_ring[i]))
                face = rgba[:3] + (0.65,)
                edge = bubble_base_edge_color
                lw_use = bubble_base_edge_lw
            else:
                base_rgba = mcolors.to_rgba(bubble_panel_a_color, bubble_panel_a_alpha)
                face = base_rgba[:3] + (bubble_panel_a_face_alpha,)
                edge_rgba = mcolors.to_rgba(bubble_panel_a_edge_color, bubble_panel_a_edge_alpha)
                edge = edge_rgba
                lw_use = bubble_panel_a_lw
            ax_a.add_patch(Ellipse((x_am_rot[i], y_am_rot[i]),
                                   width=(min_as[i]/60.0)*2.0,
                                   height=(maj_as[i]/60.0)*2.0,
                                   angle=angle_plot,
                                   facecolor=face, edgecolor=edge, lw=lw_use, zorder=bubble_panel_a_zorder))

    # 叠加 Brinks 区域标记，使用像素坐标直接转为 arcmin 后旋转
    if brinks_region_show and brinks_region_path:
        reg_path = Path(brinks_region_path)
        if reg_path.exists():
            xs_reg = []
            ys_reg = []
            pat_xy = re.compile(r"(?:point|circle)\s*\(\s*([0-9.+\-eE]+)\s*,\s*([0-9.+\-eE]+)")
            for line in reg_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("global") or line.lower().startswith("image"):
                    continue
                m = pat_xy.search(line)
                if m is None:
                    if DEBUG_PRINT_BRINKS:
                        print(f"[r_v_diagram] Skip DS9 line (no match): {line}")
                    continue
                xs_reg.append(pd.to_numeric(m.group(1), errors="coerce"))
                ys_reg.append(pd.to_numeric(m.group(2), errors="coerce"))
            if len(xs_reg) > 0:
                xs_reg = np.asarray(xs_reg, dtype=float)
                ys_reg = np.asarray(ys_reg, dtype=float)
                # DS9 像素为 1-based；转为 0-based 再相对中心取偏移
                xs_reg -= 1.0
                ys_reg -= 1.0
                x_am_reg = (xs_reg - nx_a/2.0) * cd1_am
                y_am_reg = (ys_reg - ny_a/2.0) * cd2_am
                x_am_reg_rot, y_am_reg_rot = _rotate_xy(x_am_reg, y_am_reg, rotate_bubble_pos_deg)
                if DEBUG_PRINT_BRINKS:
                    print(f"[r_v_diagram] Parsed {len(xs_reg)} Brinks DS9 entries from {reg_path}")
                    print("[r_v_diagram] Brinks DS9 raw pix (1-based):", list(zip(xs_reg + 1.0, ys_reg + 1.0)))
                    print("[r_v_diagram] Brinks arcmin offset:", list(zip(x_am_reg, y_am_reg)))
                    print("[r_v_diagram] Brinks arcmin rotated:", list(zip(x_am_reg_rot, y_am_reg_rot)))
                base_rgba = mcolors.to_rgba(brinks_region_color)
                face_rgba = (base_rgba[:3] + (brinks_region_face_alpha,)) if brinks_region_face_alpha > 0 else "none"
                edge_rgba = base_rgba[:3] + (brinks_region_edge_alpha,)
                ax_a.scatter(
                    x_am_reg_rot,
                    y_am_reg_rot,
                    marker=brinks_region_marker,
                    s=brinks_region_size,
                    facecolors=face_rgba,
                    edgecolors=edge_rgba,
                    linewidths=brinks_region_lw,
                    alpha=brinks_region_alpha,
                    zorder=brinks_region_zorder,
                )
            elif DEBUG_PRINT_BRINKS:
                print(f"[r_v_diagram] No valid Brinks DS9 entries parsed from {reg_path}")
        elif DEBUG_PRINT_BRINKS:
            print(f"[r_v_diagram] Brinks DS9 region file not found: {reg_path}")

    _shrink_row_axes_horiz([ax_a], width_frac_top)
    fig.subplots_adjust(hspace=hspace_top_mid)

    # ========== 中行 (b)(c)(d) ==========
    mid_gs = gs[1, 0].subgridspec(nrows=1, ncols=3,
                                   width_ratios=mid_width_ratios, wspace=mid_wspace)
    ax_b = fig.add_subplot(mid_gs[0, 0])
    ax_c = fig.add_subplot(mid_gs[0, 1])
    ax_d = fig.add_subplot(mid_gs[0, 2])
    ax_mid = [ax_b, ax_c, ax_d]
    title_mid = [title_b, title_c, title_d]
    show_flags = [b_show_labels, c_show_labels, d_show_labels]

    for idx, ax in enumerate(ax_mid):
        norm_pow = PowerNorm(gamma=1.9, vmin=vmin_list_center[idx], vmax=vmax_list_center[idx])
        ax.imshow(data_list_center[idx], origin='lower', cmap='gray',
                  norm=norm_pow, extent=extent_list_center[idx])
        ax.set_xlim(xlim_arcmin); ax.set_ylim(ylim_arcmin)
        ax.tick_params(axis='both', color='white', direction='in')

        # 中行刻度密度
        if mid_xtick_major is not None:
            ax.xaxis.set_major_locator(MultipleLocator(mid_xtick_major))
        if mid_ytick_major is not None:
            ax.yaxis.set_major_locator(MultipleLocator(mid_ytick_major))

        # 是否显示坐标文字
        if show_flags[idx]:
            ax.tick_params(labelbottom=True, labelleft=True, labelcolor='black')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}'"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}'"))
            ax.set_xlabel(''); ax.set_ylabel('')
        else:
            ax.tick_params(labelbottom=False, labelleft=False)

        ax.text(0.05, 0.95, title_mid[idx], transform=ax.transAxes,
                fontsize=title_fs_mid, color='white', va='top')

    _shrink_row_axes_horiz([ax_b, ax_c, ax_d], width_frac_mid)
    fig.subplots_adjust(hspace=hspace_mid_bottom)

    # ========== 底行 (e) ==========
    ax_e = fig.add_subplot(gs[2, 0])

    # r–v 参考曲线与时间线
    colors_rsn = [(0.1, 0.3, 0.7, 0.9), (0.2, 0.5, 0.8, 0.8), (0.4, 0.6, 0.9, 0.7)]
    label_x_rsn = [350, 140, 800]; label_ang_rsn = [-60, -75, -5]
    for i, (rsn, c) in enumerate(zip(rsn_list, colors_rsn)):
        v_curve = ((98392505.0 * rsn) / (r_vals**2 * n0))**(1.0/3.0)
        ax_e.plot(r_vals, v_curve, ls='-.', lw=0.5, color=c)
        lx = label_x_rsn[i]; ly = ((98392505.0 * rsn) / (lx**2 * n0))**(1.0/3.0)
        ax_e.text(lx, ly, f"{rsn} SN Myr$^{{-1}}$",
                  fontsize=8, color=c, rotation=label_ang_rsn[i],
                  rotation_mode='anchor', ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0.3))

    colors_t = [(0.7, 0.2, 0.2, 0.7), (0.9, 0.5, 0.2, 0.7), (0.6, 0.6, 0.6, 0.7)]
    label_x_t = [600, 930, 790]; label_ang_t = [61, 43, 27]
    for i, (T, c) in enumerate(zip(T_list, colors_t)):
        v_T = 0.6 * r_vals / T
        ax_e.plot(r_vals, v_T, ls='-', lw=0.5, color=c)
        lx = label_x_t[i]; ly = 0.6 * lx / T
        ax_e.text(lx, ly, f"{T} Myr", fontsize=8, color=c,
                  rotation=label_ang_t[i], rotation_mode='anchor',
                  ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0.3))

    # (e) 误差棒（相对误差）
    if color_by_nring:
        data_colors = np.asarray([plt.get_cmap(color_cmap)(norm_n(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 1.0) for v in n_ring])
    else:
        base_rgba = bubble_base_face_color if bubble_base_face_color is not None else (0.5, 0.5, 0.8, 0.5)
        data_colors = np.repeat(np.array(base_rgba)[None, :], len(n_ring), axis=0)
    if bubble_show_errorbar:
        for xi, yi, col in zip(r_pc, v_exp, data_colors):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            xe = abs(err_frac_x) * abs(xi)
            ye = abs(err_frac_y) * abs(yi)
            ecolor = (col[0], col[1], col[2], float(bubble_errorbar_data_alpha)) \
                     if bubble_errorbar_use_data_color else (0, 0, 0, float(bubble_errorbar_data_alpha))
            ax_e.errorbar(
                [xi], [yi],
                xerr=[[xe], [xe]], yerr=[[ye], [ye]],
                fmt='none',
                ecolor=ecolor,
                elinewidth=bubble_errorbar_elinewidth,
                capsize=bubble_errorbar_capsize,
                ls=bubble_errorbar_linestyle,
                zorder=bubble_errorbar_zorder
            )

    # (e) 氣泡点（常规着色）
    facecolors = np.c_[data_colors[:, :3], np.full(len(data_colors), float(bubble_marker_face_alpha))]
    edge_rgba_base = mcolors.to_rgba(bubble_marker_edge_color)
    alpha_scale = 1.0 if bubble_marker_edge_alpha is None else float(bubble_marker_edge_alpha)
    edge_rgba = edge_rgba_base[:3] + (edge_rgba_base[3] * alpha_scale,)
    edgecolors = np.repeat(np.array(edge_rgba)[None, :], len(facecolors), axis=0)
    bubble_handle = ax_e.scatter(
        r_pc, v_exp,
        s=bubble_marker_size,
        facecolor=facecolors,
        edgecolor=None,
        linewidths=0,
        zorder=max(bubble_marker_zorder, 9)
    )

    # (e) 叠加 Brinks+86 数据点（与 (a) 样式保持一致，可单独调节）
    brinks_handle = None
    if brinks_panel_e_show and brinks_table_path:
        bpath = Path(brinks_table_path)
        if bpath.exists():
            bdf = pd.read_fwf(bpath)
            required_cols = ["Diam_pc", "DV_kms"]
            missing_cols = [c for c in required_cols if c not in bdf.columns]
            if missing_cols:
                if DEBUG_PRINT_BRINKS:
                    print(f"[r_v_diagram] Missing columns in Brinks table {bpath}: {missing_cols}")
            else:
                diam_pc_b = pd.to_numeric(bdf["Diam_pc"], errors="coerce").to_numpy(dtype=float)
                dv_kms_b = np.abs(pd.to_numeric(bdf["DV_kms"], errors="coerce").to_numpy(dtype=float))
                r_pc_b = 0.5 * diam_pc_b
                vexp_b = dv_kms_b
                mask_b = np.isfinite(r_pc_b) & np.isfinite(vexp_b)
                if np.any(mask_b):
                    marker_use = brinks_panel_e_marker if (brinks_panel_e_marker is not None) else brinks_region_marker
                    size_use = brinks_panel_e_size if (brinks_panel_e_size is not None) else brinks_region_size
                    color_use = brinks_panel_e_color if (brinks_panel_e_color is not None) else brinks_region_color
                    edge_alpha_use = brinks_panel_e_edge_alpha if (brinks_panel_e_edge_alpha is not None) else brinks_region_edge_alpha
                    face_alpha_use = brinks_panel_e_face_alpha if (brinks_panel_e_face_alpha is not None) else brinks_region_face_alpha
                    lw_use = brinks_panel_e_lw if (brinks_panel_e_lw is not None) else brinks_region_lw
                    alpha_use = brinks_panel_e_alpha if (brinks_panel_e_alpha is not None) else brinks_region_alpha
                    zorder_use = brinks_panel_e_zorder if (brinks_panel_e_zorder is not None) else brinks_region_zorder
                    base_rgba = mcolors.to_rgba(color_use)
                    face_rgba = base_rgba[:3] + (face_alpha_use,) if face_alpha_use > 0 else "none"
                    edge_rgba = base_rgba[:3] + (edge_alpha_use,)
                    brinks_handle = ax_e.scatter(
                        r_pc_b[mask_b],
                        vexp_b[mask_b],
                        marker=marker_use,
                        s=size_use,
                        facecolors=face_rgba,
                        edgecolors=edge_rgba,
                        linewidths=lw_use,
                        alpha=alpha_use,
                        zorder=zorder_use
                    )
                elif DEBUG_PRINT_BRINKS:
                    print(f"[r_v_diagram] No finite Brinks rows in {bpath}")
        elif DEBUG_PRINT_BRINKS:
            print(f"[r_v_diagram] Brinks table not found: {bpath}")

    # Legends for panel (e)
    legend_items = []
    legend_labels = []
    if bubble_handle is not None:
        legend_items.append(bubble_handle)
        legend_labels.append("New Bubbles")
    if brinks_handle is not None:
        legend_items.append(brinks_handle)
        legend_labels.append("Brinks+86")
    if legend_items:
        ax_e.legend(legend_items, legend_labels, loc='upper right', fontsize=8, frameon=True, framealpha=0.8)

    _shrink_row_axes_horiz([ax_e], width_frac_bottom)

    ax_e.set_xlabel(r'$r$ [pc]')
    ax_e.set_ylabel(r'$v_{\rm exp}$ [km s$^{-1}$]')
    if e_xtick_major is not None:
        ax_e.xaxis.set_major_locator(MultipleLocator(e_xtick_major))
    if e_ytick_major is not None:
        ax_e.yaxis.set_major_locator(MultipleLocator(e_ytick_major))
    ax_e.set_xlim([0, 1000]); ax_e.set_ylim([0, 40])
    ax_e.tick_params(direction='in')
    ax_e.text(0.02, 0.97, title_e, transform=ax_e.transAxes, fontsize=title_fs_e, va='top')

    # ---------- 高亮（记录真实坐标；ring/star/emph 三种样式） ----------
    star_a_xy = None  # (a) 高亮在 (a) 的 arcmin 坐标
    star_e_xy = None  # (e) 高亮在 (e) 的 (pc, km/s) 坐标

    def _draw_highlight_marker(ax, x, y):
        style = str(highlight_style).lower()
        if style == 'star':
            ax.scatter([x], [y], marker='*', s=hl_marker_size,
                       facecolor=hl_star_facecolor, edgecolor=hl_star_edgecolor,
                       linewidth=hl_star_linewidth, zorder=12)
        elif style == 'ring':
            base_s = float(hl_marker_size)
            ax.scatter([x], [y], marker='o', s=base_s*(hl_ring_outer_scale**2),
                       facecolors='none', edgecolors=hl_ring_outer_edgecolor,
                       linewidths=hl_ring_outer_linewidth, zorder=12)
            ax.scatter([x], [y], marker='o', s=base_s,
                       facecolors='none', edgecolors=hl_ring_edgecolor,
                       linewidths=hl_ring_linewidth, zorder=13)
            ax.scatter([x], [y], marker='o', s=base_s*(hl_ring_inner_scale**2),
                       facecolors='none', edgecolors=hl_ring_inner_edgecolor,
                       linewidths=hl_ring_inner_linewidth, zorder=14)
        else:
            # emph 在 (e) 上的点强调（(a) 的椭圆已在上面绘制时处理）
            ax.scatter([x], [y], s=_e_size,
                       facecolor=_e_face, edgecolor=_e_edge,
                       linewidth=_e_lw, zorder=12)

    # 找到高亮行
    row_hl = None
    if highlight_id is not None and has_id:
        sel = df.loc[df['id'] == highlight_id]
        if len(sel) > 0:
            row_hl = sel.iloc[0]

    if row_hl is not None:
        # (a) 坐标（用于 ring/star；emph 的椭圆已在椭圆循环内处理完成）
        coord_hl = SkyCoord(str(row_hl['ra_hms']), str(row_hl['dec_dms']),
                            unit=(u.hourangle, u.deg))
        xh_pix, yh_pix = wcs_a.world_to_pixel(coord_hl)
        xh_am = (xh_pix - nx_a/2.0) * cd1_am
        yh_am = (yh_pix - ny_a/2.0) * cd2_am
        xh_am_r, yh_am_r = _rotate_xy(xh_am, yh_am, rotate_bubble_pos_deg)
        star_a_xy = (float(xh_am_r), float(yh_am_r))

        # (e) 坐标
        r_hl = float(row_hl['radius_pc'])
        v_hl = float(row_hl['expansion_vel'])
        star_e_xy = (r_hl, v_hl)

        # 绘制高亮标记
        if str(highlight_style).lower() in ('ring', 'star'):
            _draw_highlight_marker(ax_a, star_a_xy[0], star_a_xy[1])
        _draw_highlight_marker(ax_e, star_e_xy[0], star_e_xy[1])

    # —— 先紧布局，固定各 panel 位置 —— 
    plt.tight_layout()

    # ========== 在 panel e 右侧加 colorbar（使用紧布局后的最终位置） ==========
    if color_by_nring:
        bbox = ax_e.get_position()
        if isinstance(e_colorbar_size, str) and e_colorbar_size.endswith('%'):
            frac = float(e_colorbar_size.strip('%')) / 100.0
            cax_width = bbox.width * frac
        else:
            cax_width = float(e_colorbar_size)
        cax_pad = bbox.width * float(e_colorbar_pad)
        cax_x0 = bbox.x1 + cax_pad
        cax = fig.add_axes([cax_x0, bbox.y0, cax_width, bbox.height])

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(color_cmap), norm=norm_n)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(r"$n_{\rm HI}$ [cm$^{-3}$]")
        ticks = np.linspace(float(color_vmin), float(color_vmax), 8)
        def _fmt_tick(v):
            return f"{v:.2f}".rstrip('0').rstrip('.')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([_fmt_tick(v) for v in ticks])

    # ---------- 连线工具 ----------
    def data_to_fig_coords(ax, x, y):
        return ax.transData.transform((x, y))
    def disp_to_fig(xy_disp):
        return fig.transFigure.inverted().transform(xy_disp)
    def draw_link(src_ax, src_xy, tgt_ax, tgt_xy, color, lw, ls, alpha):
        if (src_xy is None) or (tgt_xy is None):
            return
        try:
            sx, sy = src_xy
            tx, ty = tgt_xy
            src_fig = disp_to_fig(data_to_fig_coords(src_ax, sx, sy))
            tgt_fig = disp_to_fig(data_to_fig_coords(tgt_ax, tx, ty))
            fig.lines.append(plt.Line2D([src_fig[0], tgt_fig[0]],
                                        [src_fig[1], tgt_fig[1]],
                                        transform=fig.transFigure,
                                        color=color, lw=lw, ls=ls, alpha=alpha, zorder = 1))
        except Exception as e:
            print(f"[WARN] 连线失败: {e}")

    # ---------- 目标角点（中排改为 b/d） ----------
    xlim_b = ax_b.get_xlim(); ylim_b = ax_b.get_ylim()
    xlim_c = ax_c.get_xlim(); ylim_c = ax_c.get_ylim()  # 仍用于可能的手动覆盖
    xlim_d = ax_d.get_xlim(); ylim_d = ax_d.get_ylim()
    b_ul = (xlim_b[0], ylim_b[1]); b_ll = (xlim_b[0], ylim_b[0])
    d_ur = (xlim_d[1], ylim_d[1]); d_lr = (xlim_d[1], ylim_d[0])

    # 小工具：叠加偏移
    def _add_delta(xy, delta):
        if (xy is None) or (delta is None):
            return xy
        dx, dy = float(delta[0]), float(delta[1])
        return (xy[0] + dx, xy[1] + dy)

    # 解析：默认/覆盖 + 偏移
    def _resolve_link(default_src, default_tgt, enabled,
                      src_override, src_delta, tgt_override, tgt_delta,
                      ls, color, lw, alpha, src_ax, tgt_ax):
        if not enabled:
            return None
        sxy = tuple(src_override) if (src_override is not None) else default_src
        txy = tuple(tgt_override) if (tgt_override is not None) else default_tgt
        if (sxy is None) or (txy is None):
            return None
        sxy = _add_delta(sxy, src_delta)
        txy = _add_delta(txy, tgt_delta)
        return (sxy, txy, ls, color, lw, alpha, src_ax, tgt_ax)

    # ---------- 解析“手动覆盖/默认”并画线 ----------
    links_to_draw = []
    if auto_corner_links:
        # a→b 左上 / a→d 右上
        links_to_draw.append(_resolve_link(star_a_xy, b_ul,
            True, None, link_a2c_ul_src_delta,
            None, link_a2c_ul_tgt_delta,
            link_a2c_ul_ls, link_a2c_ul_color, link_a2c_ul_lw, link_a2c_ul_alpha,
            ax_a, ax_b))
        links_to_draw.append(_resolve_link(star_a_xy, d_ur,
            True, None, link_a2c_ur_src_delta,
            None, link_a2c_ur_tgt_delta,
            link_a2c_ur_ls, link_a2c_ur_color, link_a2c_ur_lw, link_a2c_ur_alpha,
            ax_a, ax_d))
        # e→b 左下 / e→d 右下
        links_to_draw.append(_resolve_link(star_e_xy, b_ll,
            True, None, link_e2c_ll_src_delta,
            None, link_e2c_ll_tgt_delta,
            link_e2c_ll_ls, link_e2c_ll_color, link_e2c_ll_lw, link_e2c_ll_alpha,
            ax_e, ax_b))
        links_to_draw.append(_resolve_link(star_e_xy, d_lr,
            True, None, link_e2c_lr_src_delta,
            None, link_e2c_lr_tgt_delta,
            link_e2c_lr_ls, link_e2c_lr_color, link_e2c_lr_lw, link_e2c_lr_alpha,
            ax_e, ax_d))
    else:
        pass

    # 实际绘制
    for idx, item in enumerate(links_to_draw):
        if item is None:
            continue
        sxy, txy, ls, color, lw, alpha, src_ax, tgt_ax = item
        draw_link(src_ax, sxy, tgt_ax, txy, color=color, lw=lw, ls=ls, alpha=alpha)

    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


# ====== 示例调用 ======
if __name__ == '__main__':
    plot_five_panels_three_rows(
        # 顶行 (a)
        fits_rotated_a='../data/jcomb_submed.mom0.-600-45_rotated.fits',
        table_path='../code/1113.tab',
        a_norm_kind='power', a_power_gamma=0.6,
        a_vmin=0.1, a_vmax=10,
        a_xlim_arcmin=(-110, 110), a_ylim_arcmin=(-40, 40),
        direction_angle_a=-52,
        rot_delta_deg=52.0,
        rotate_bubble_pos_deg=52.0,
        dist_mpc=0.761, scalebar_kpc=5.0,

        # 中行 (b)(c)(d)
        fits_path_list_center=[
            '../data/b302_-94-88.fits',
            '../data/b302_-88-82.fits',
            '../data/b302_-82-76.fits'
        ],
        xlim_arcmin=(-5, 5), ylim_arcmin=(-5, 5),
        vmin_list_center=(0.01, 0.01, 0.01),
        vmax_list_center=(4.0, 4.0, 4.0),

        # 标题与字号
        title_a='(a)',
        title_b=r'(b) $(-94,-88)$',
        title_c=r'(c) $(-88,-82)$',
        title_d=r'(d) $(-82,-76)$',
        title_e='(e)',
        title_fs_a=11, title_fs_mid=9, title_fs_e=11,

        # 中行坐标文字显示选择
        b_show_labels=True, c_show_labels=False, d_show_labels=False,

        # 刻度密度
        a_xtick_major=50.0, a_ytick_major=20.0,
        mid_xtick_major=2.0,  mid_ytick_major=2.0,
        e_xtick_major=None,   e_ytick_major=None,

        # 底行 (e)：连续着色与点样式（color_vmin/color_vmax 默认自动取）
        color_cmap='coolwarm',
        r_curve_minmax=(90, 2000), n0=0.2,
        rsn_list=(10, 1, 0.1), T_list=(10, 20, 40),

        bubble_marker_size=65,
        bubble_marker_face_alpha=0.40,
        bubble_marker_edge_lw=0.8,
        bubble_marker_edge_alpha=0.95,
    bubble_marker_edge_color=(0.9, 0.2, 0.2),
        bubble_marker_zorder=6,

        bubble_show_errorbar=True,
        bubble_errorbar_use_data_color=True,
        bubble_errorbar_data_alpha=0.0,
        bubble_errorbar_elinewidth=0.6,
        bubble_errorbar_capsize=0.0,
        bubble_errorbar_linestyle='-',
        bubble_errorbar_zorder=5,
        err_frac_x=0.25, err_frac_y=0.25,

        # 高亮（选择 'ring' / 'star' / 'emph'）
        highlight_id=104,
        highlight_style='emph',
        hl_marker_size=45,

        hl_emph_a_facecolor=(1.0, 0.9, 0.2, 0.95),
        hl_emph_a_edgecolor='k',
        hl_emph_a_ellipse_lw=1.0,

        hl_emph_e_facecolor=(1.0, 0.9, 0.2, 0.95),
        hl_emph_e_edgecolor='k',
        hl_emph_e_point_size=75,
        hl_emph_e_point_edge_lw=1.0,

        auto_corner_links=True,

        figsize=(4.5, 7.5),
        height_ratios=(1.2, 1.0, 3.0),
        hspace_top_mid=-0.9,
        hspace_mid_bottom=-0.2,
        mid_wspace=0.,
        mid_width_ratios=(1.0, 1.0, 1.0),
        e_colorbar_size='4.5%',
        e_colorbar_pad=0.01,
        width_frac_top=1.0,
        width_frac_mid=1.0,
        width_frac_bottom=1.0,

        output_name='r_v_diagram.pdf'
    )
