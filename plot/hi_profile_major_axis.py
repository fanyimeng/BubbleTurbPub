import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import FuncNorm
import matplotlib.pyplot as plt

def plot_full_panel(
    fits_path,
    profile_csv,
    output_name='hi_profile_major_axis.pdf',
    xlim_kpc=(5.0, 21.0),
    ylim_pix=(1300, 1700),
    vmin=None,
    vmax=None,
    order=['top0', 'top1', 'middle', 'bottom'],
    ref_line_range=(5.0, 21.0),
    # 可自定义纵轴范围
    ylims_top0=(0.003, 5.0),     
    ylims_top1=(0.0, 38.0),      
    ylims_middle=(0.0, 700.0),   
    bottom_ylim_kpc=None,        
    # 可自定义 legend 样式与位置
    legend_top0=None,            
    legend_top1=None,            
    # 可自定义 legend 的文字
    labels_top0=None,            
    labels_top1=None             
):
    hdu = fits.open(fits_path)[0]
    data = hdu.data
    header = hdu.header

    # 显示用缩放
    data = data * 1.222e6 / 1.4**2 / 60 / 60 * 1.8e18 / 1e20

    arcmin_per_pix = np.abs(header['CDELT1']) * 60.0
    pix2kpc = arcmin_per_pix * 216 / 1000.0

    x0 = int(xlim_kpc[0] / pix2kpc) + 1500
    x1_ = int(xlim_kpc[1] / pix2kpc) + 1500
    y0, y1 = ylim_pix
    cutout = data[y0:y1, x0:x1_]
    ny, nx = cutout.shape
    extent = [xlim_kpc[0], xlim_kpc[1], 0, ny * pix2kpc]

    # 读 radial profile
    df = pd.read_fwf(profile_csv, infer_nrows=int(1e6))
    r_kpc = df['r'].values / 1000.0
    mom0_x1 = df['x1_mom0'].values
    mom2_x1 = df['x1_mom2'].values
    mom0_x2 = df['x2_mom0'].values
    mom2_x2 = df['x2_mom2'].values
    mom2_x3 = df['x3_mom2'].values if 'x3_mom2' in df.columns else None

    cosi = np.cos(np.deg2rad(77))
    xx = r_kpc
    yy  = xx * 16 + 182
    yy1 = xx * 19 + 182 + 37
    yy2 = xx * 13 + 182 - 37

    def mom0_to_nHI(mom0_arr, h_pc):
        NHI = mom0_arr * 1.222e6 / 1.4**2 / 60 / 60 * 1.8e18
        nHI = NHI / (2 * h_pc / cosi * 3e18)
        return nHI

    mom0_mid_x1  = mom0_to_nHI(mom0_x1, yy)
    mom0_high_x1 = mom0_to_nHI(mom0_x1, yy2)
    mom0_low_x1  = mom0_to_nHI(mom0_x1, yy1)
    mom0_mid_x2  = mom0_to_nHI(mom0_x2, yy)
    mom0_high_x2 = mom0_to_nHI(mom0_x2, yy2)
    mom0_low_x2  = mom0_to_nHI(mom0_x2, yy1)

    mask = (r_kpc >= ref_line_range[0]) & (r_kpc <= ref_line_range[1])
    r_kpc_plot = r_kpc[mask]
    mom0_mid_x1_plot  = mom0_mid_x1[mask]
    mom0_high_x1_plot = mom0_high_x1[mask]
    mom0_low_x1_plot  = mom0_low_x1[mask]
    mom0_mid_x2_plot  = mom0_mid_x2[mask]
    mom0_high_x2_plot = mom0_high_x2[mask]
    mom0_low_x2_plot  = mom0_low_x2[mask]
    mom2_x1_plot = mom2_x1[mask]
    mom2_x2_plot = mom2_x2[mask]
    mom2_x3_plot = mom2_x3[mask] if mom2_x3 is not None else None

    height_dict = {'top0': 0.2, 'top1': 0.2, 'middle': 0.2, 'bottom': 0.4}
    bottom_pos  = {'top0': 0.73, 'top1': 0.53, 'middle': 0.33, 'bottom': 0.0}
    fig = plt.figure(figsize=(6, 6))
    axs = {key: fig.add_axes([0.13, bottom_pos[key], 0.80, height_dict[key]]) for key in order}

    col_x1 = 'black'
    shade_x1 = (0.4, 0.4, 0.4, 0.35)
    col_x2 = 'tab:blue'
    shade_x2 = (0.25, 0.45, 0.95, 0.28)

    if labels_top0 is None:
        labels_top0 = {'x1_mid': 'x1 mid','x1_range': 'x1 range',
                       'x2_mid': 'x2 mid','x2_range': 'x2 range'}
    if labels_top1 is None:
        labels_top1 = {'x1_turb': 'x1 turb','x1_therm': 'x1 thermal',
                       'x2_turb': 'x2 turb','x2_therm': 'x2 thermal'}

    # (a) HI 数密度
    axs['top0'].plot(r_kpc_plot, mom0_mid_x1_plot, color=col_x1, lw=1.2,
                     label=labels_top0['x1_mid'])
    axs['top0'].fill_between(r_kpc_plot, mom0_low_x1_plot, mom0_high_x1_plot,
                             color=shade_x1)
    axs['top0'].plot(r_kpc_plot, mom0_mid_x2_plot, color=col_x2, lw=1.2, ls='--',
                     label=labels_top0['x2_mid'])
    axs['top0'].fill_between(r_kpc_plot, mom0_low_x2_plot, mom0_high_x2_plot,
                             color=shade_x2)
    axs['top0'].set_xlim(xlim_kpc)
    axs['top0'].set_ylim(*ylims_top0)
    axs['top0'].set_ylabel('$n_{\\rm HI}$ [cm$^{-3}$]')
    axs['top0'].set_yscale('log')
    axs['top0'].tick_params(axis='x', labelbottom=False)
    axs['top0'].tick_params(axis='y', direction='in')
    # 🚫 去掉 y 轴 minor ticks
    axs['top0'].tick_params(axis='y', which='minor', left=False, right=False)
    axs['top0'].text(0.01, 0.90, '(a)', transform=axs['top0'].transAxes,
                     fontsize=10, color='black', va='top')
    if legend_top0 is None:
        legend_top0 = dict(loc='upper right', fontsize=8, frameon=False, ncol=2)
    axs['top0'].legend(**legend_top0)

    # (b) 涡动速度
    def subtract_thermal(m2_arr, thermal_kms=8.1):
        return np.sqrt(np.maximum(0.0, m2_arr**2 - thermal_kms**2))
    m2_turb_x1 = subtract_thermal(mom2_x1_plot)
    #     col_x1 = 'black'
    # shade_x1 = (0.4, 0.4, 0.4, 0.35)
    axs['top1'].plot(r_kpc_plot, m2_turb_x1, color=col_x1, ls='-',
                     label=labels_top1['x1_turb'])
    axs['top1'].fill_between(r_kpc_plot, m2_turb_x1, mom2_x1_plot,
                             color=shade_x1,
                             label=labels_top1['x1_therm'])
    m2_turb_x2 = subtract_thermal(mom2_x2_plot)
    axs['top1'].plot(r_kpc_plot, m2_turb_x2, color=col_x2, ls='--',
                     label=labels_top1['x2_turb'])
    axs['top1'].fill_between(r_kpc_plot, m2_turb_x2, mom2_x2_plot,
                             color=shade_x2,
                             label=labels_top1['x2_therm'])
    axs['top1'].set_xlim(xlim_kpc)
    axs['top1'].set_ylim(*ylims_top1)
    axs['top1'].set_ylabel('$\\sigma_{\\rm Turb}$ [km/s]')
    axs['top1'].tick_params(axis='x', labelbottom=False)
    axs['top1'].tick_params(direction='in')
    axs['top1'].text(0.01, 0.90, '(b)', transform=axs['top1'].transAxes,
                     fontsize=10, color='black', va='top')
    if legend_top1 is None:
        legend_top1 = dict(loc='upper right', fontsize=8, frameon=False, ncol=2)
    axs['top1'].legend(**legend_top1)

    # (c) 厚度
    xx_h = np.arange(ref_line_range[0], ref_line_range[1], 0.1)
    yy_h, yy1_h, yy2_h = xx_h*16+182, xx_h*19+182+37, xx_h*13+182-37
    axs['middle'].plot(xx_h, yy_h, color=col_x1)
    axs['middle'].fill_between(xx_h, yy1_h, yy2_h, color=shade_x1)
    axs['middle'].set_xlim(xlim_kpc)
    axs['middle'].set_ylim(*ylims_middle)
    axs['middle'].set_ylabel('h [pc]')
    axs['middle'].tick_params(axis='x', labelbottom=False)
    axs['middle'].tick_params(direction='in')
    axs['middle'].text(0.01, 0.90, '(c)', transform=axs['middle'].transAxes,
                       fontsize=10, color='black', va='top')

    # (d) 图像
    norm = FuncNorm((np.arcsinh, np.sinh), vmin=vmin, vmax=vmax)
    axs['bottom'].imshow(cutout, origin='lower', cmap='RdYlBu_r', norm=norm, extent=extent)
    axs['bottom'].set_xlabel('x [kpc]')
    if bottom_ylim_kpc is not None:
        axs['bottom'].set_ylim(*bottom_ylim_kpc)
    axs['bottom'].set_yticks([])
    axs['bottom'].tick_params(axis='x', direction='in')
    axs['bottom'].tick_params(axis='y', length=0)
    y_display = (1500 - y0) * pix2kpc
    axs['bottom'].plot([ref_line_range[0], ref_line_range[1]], [y_display, y_display],
                       color='black', lw=2, ls='--')
    axs['bottom'].text(0.01, 0.97, '(d)', transform=axs['bottom'].transAxes,
                       fontsize=10, color='black', va='top')

    plt.savefig(output_name, dpi=200, bbox_inches='tight')
    plt.close()


# === 调用示例 ===
plot_full_panel(
    fits_path='../data/bubbleTurbCube_x1beam_-700-100_0.006_mom0_rotated.fits',
    profile_csv='../data/profile_output_VLA_JCOMB.txt',
    output_name='hi_profile_major_axis.pdf',
    xlim_kpc=(2.0, 21.5),
    ylim_pix=(1350, 1650),
    vmin=1,
    vmax=100,
    ylims_top0=(0.002, 6.0),
    ylims_top1=(0.0, 57.0),
    ylims_middle=(0.0, 750.0),
    legend_top0=dict(loc='upper right', fontsize=8, frameon=False, ncol=2),
    legend_top1=dict(loc='upper right', fontsize=8, frameon=False, ncol=2),
    labels_top0={
        'x1_mid': 'FAST+VLA',
        'x1_range': 'x1 envelope',
        'x2_mid': 'VLA',
        'x2_range': 'x2 envelope'
    },
    labels_top1={
        'x1_turb': 'FAST+VLA turb.',
        'x1_therm': 'FAST+VLA therm.',
        'x2_turb': 'VLA turb.',
        'x2_therm': 'VLA therm.'
    }
)
