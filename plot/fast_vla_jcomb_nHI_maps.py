import numpy as np
from astropy.io import fits
from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt

def plot_three_panels_arcmin_bottom_labels(
    fits_path_list,
    output_name='fast_vla_jcomb_nHI_maps.pdf',
    xlim_arcmin=(-180, 180),
    ylim_arcmin=(-60, 60),
    vmin_list=None,
    vmax_list=None,
    colorbar_pad=0.005,
    panel_labels=None,
    direction_angle=52  # 默认 52°，竖直向上=0°，逆时针正
):
    assert len(fits_path_list) == 3, "必须提供三个 FITS 文件路径"
    if vmin_list is None:
        vmin_list = [None, None, None]
    if vmax_list is None:
        vmax_list = [None, None, None]
    if panel_labels is None:
        panel_labels = ['a. FAST', 'b. VLA', 'c. FAST+VLA']

    freq_GHz = 1.4

    # === Step 1: 打开 FITS 数据并计算 column density ===
    data_list = []
    extent_list = []
    for fits_path in fits_path_list:
        hdu = fits.open(fits_path)[0]
        data = hdu.data
        header = hdu.header

        if 'BMAJ' not in header:
            raise ValueError(f"FITS 文件 {fits_path} 缺少 BMAJ")
        beam_as = header['BMAJ'] * 3600.0  # arcsec

        data = data * 1.222e6 / (freq_GHz**2 * beam_as**2) * 1.8e18 / 1e20

        nx = header['NAXIS1']
        ny = header['NAXIS2']
        cdelt1 = np.abs(header['CDELT1']) * 60.0
        cdelt2 = np.abs(header['CDELT2']) * 60.0
        x_center = nx / 2.0
        y_center = ny / 2.0

        x_extent = (np.arange(nx) - x_center) * cdelt1
        y_extent = (np.arange(ny) - y_center) * cdelt2
        extent = [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]]

        data_list.append(data)
        extent_list.append(extent)

    # === Step 2: 创建 Figure 和面板 ===
    fig = plt.figure(figsize=(6, 6.3))
    panel_height = 0.28
    bottom_positions = [0.60, 0.32, 0.04]
    axs = []

    for i in range(3):
        ax = fig.add_axes([0.13, bottom_positions[i], 0.80, panel_height])
        axs.append(ax)

    # === Step 3: 绘制三个面板 ===
    im_list = []
    for idx, ax in enumerate(axs):
        norm = PowerNorm(gamma=0.6, vmin=vmin_list[idx], vmax=vmax_list[idx])
        im = ax.imshow(data_list[idx], origin='lower', cmap='afmhot',
                       norm=norm, extent=extent_list[idx])
        im_list.append(im)

        ax.set_xlim(xlim_arcmin)
        ax.set_ylim(ylim_arcmin)
        ax.tick_params(axis='both', color='white', direction='in')

        if idx == 2:
            ax.tick_params(labelbottom=True, labelleft=True, labelcolor='black')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}'"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}'"))
            ax.set_xlabel('')
            ax.set_ylabel('')

            x_min, x_max = xlim_arcmin
            y_min, y_max = ylim_arcmin
            arrow_base_x = x_min + 0.11 * (x_max - x_min)
            arrow_base_y = y_min + 0.08 * (y_max - y_min)
            arrow_len = 0.04 * (x_max - x_min)

            theta = np.radians(direction_angle)
            dx_N = arrow_len * np.sin(theta)
            dy_N = arrow_len * np.cos(theta)
            ax.arrow(arrow_base_x, arrow_base_y, dx_N, dy_N,
                     head_width=1, head_length=1, fc='white', ec='white', linewidth=0.7)
            ax.text(arrow_base_x + dx_N * 1.70, arrow_base_y + dy_N * 1.70, 'N',
                    color='white', fontsize=8, ha='center', va='center')

            theta_W = theta + np.pi / 2
            dx_W = arrow_len * np.sin(theta_W)
            dy_W = arrow_len * np.cos(theta_W)
            ax.arrow(arrow_base_x, arrow_base_y, dx_W, dy_W,
                     head_width=1, head_length=1, fc='white', ec='white', linewidth=0.7)
            ax.text(arrow_base_x + dx_W * 1.70, arrow_base_y + dy_W * 1.70, 'W',
                    color='white', fontsize=8, ha='center', va='center')
        else:
            ax.tick_params(labelbottom=False, labelleft=False)

        ax.text(0.02, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=10, color='white', va='top')

    # === Step 4: Colorbar ===
    box = axs[-1].get_position()
    cax_position = [box.x1 + colorbar_pad, box.y0, 0.02, box.height]
    cax = fig.add_axes(cax_position)
    cbar = fig.colorbar(im_list[-1], cax=cax, orientation='vertical')
    cbar.set_label('$N_{\\rm HI}$ [$10^{20}\\,\\rm cm^{-2}$]')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(direction='in', length=7)

    plt.savefig(output_name, dpi=200, bbox_inches='tight')
    plt.close()

# ✅ 调用示例
plot_three_panels_arcmin_bottom_labels(
    fits_path_list=[
        '../data/fast_mom0_rotated.fits',
        '../data/vla_submed.mom0.-600-45_rotated.fits',
        '../data/jcomb_submed.mom0.-600-45_rotated.fits'
    ],
    output_name='fast_vla_jcomb_nHI_maps.pdf',
    xlim_arcmin=(-110, 110),
    ylim_arcmin=(-40, 40),
    vmin_list=np.array([0.5, 0.75, 0.5]) * 0.8,
    vmax_list=np.array([60, 90, 60]) * 1.0,
    panel_labels=['a. FAST', 'b. VLA', 'c. FAST+VLA'],
    direction_angle=-52
)