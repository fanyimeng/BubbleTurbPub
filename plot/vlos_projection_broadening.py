#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-panel figure with ONE continuous colorbar (from panel b), plus dashed iso-lines on panel b.

(a) Left: Analytic model |Δv_LOS|(phi, R) using the SAME cmap/norm as panel (b).
(b) Right: Observed |Δv_los| with N_HI contours and dashed iso-|Δv_los| lines.

Colorbar is continuous (no steps) and shown ONLY for panel (b).
Layout is adjustable at call time.
Output PDF name is fixed to 'vlos_projection_broadening.pdf'.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# ---------- helpers ----------
def mom0_to_nhi_cm2(mom0_data):
    """Convert moment-0 (K km/s) to N_HI (cm^-2)."""
    return mom0_data * 1.222e6 / (1.4**2) / 60.0 / 60.0 * 1.8e18

def _angle_map(shape, center_pix, pa_deg=0.0):
    """Position angle map in degrees measured CCW from +x after subtracting pa_deg."""
    h, w = shape
    y_idx, x_idx = np.indices((h, w))
    xc, yc = center_pix  # (x, y)
    theta = np.degrees(np.arctan2(y_idx - yc, x_idx - xc))
    theta = (theta - pa_deg + 360.0) % 360.0
    return theta

def _in_angle_range(theta_deg, amin, amax):
    """Inclusive angular selection that correctly handles wrap-around."""
    theta = theta_deg % 360.0
    amin = amin % 360.0
    amax = amax % 360.0
    if amin <= amax:
        return (theta >= amin) & (theta <= amax)
    else:
        return (theta >= amin) | (theta <= amax)

def _radial_map(shape, center_pix):
    h, w = shape
    y_idx, x_idx = np.indices((h, w))
    xc, yc = center_pix
    return np.hypot(x_idx - xc, y_idx - yc)

# ---------- main plotting ----------
def plot_ab_panels_shared_cbar_continuous(
    # ---- left panel (analytic model) ----
    v_rot=220.0,
    incl_deg=77.0,
    theta_deg_max=90.0,
    theta_steps=30,
    R_min=5.0,
    R_max=30.0,
    R_steps=20,
    # ---- right panel (observed map, DEFINES the shared cmap/norm) ----
    vlos_fits_path="../data/vlos_delta_abs.fits",
    mom0_fits_path="../data/jcomb_submed.mom0.-600-45.fits",
    xlim=(450, 2550),
    ylim=(300, 2700),
    cmap_b="RdYlBu_r",
    vmin_b=0.0,
    vmax_b=40.0,
    # dashed iso-|Δvlos| on panel (b)
    sectors=None,                 # [{'theta_center':..., 'half_width':...}, ...] or [{'theta_min':..., 'theta_max':...}]
    special_levels=(1.7,),        # km/s
    special_color='w',
    special_linestyle="--",
    special_linewidth=1.0,
    special_edge=True,
    special_edge_color="k",
    special_edge_width=2.0,
    special_center_pix=None,      # (x, y) in pixels; default = center of image
    special_limit_to_vlos_mask=True,
    special_clip_to_limits=True,
    special_use_abs=True,
    special_pa_deg=38.0,
    # NHI contours
    nhi_levels=(5e20, 2e21),
    nhi_mask_min=5e20,
    contour_white_with_black_edge=True,
    contour_edge_linewidth=1.5,
    contour_linewidth=0.8,
    # ---- colorbar ticks (continuous) ----
    cbar_ticks=None,              # e.g., [0,10,20,30,40]; if None, auto from vmin_b/vmax_b
    # ---- layout tuning (relative widths; all are normalized internally) ----
    fig_size=(12.0, 4.8),
    panel_a_width=1.0,
    panel_gap=0.12,
    panel_b_width=1.0,
    cbar_gap_b=0.06,
    cbar_width=0.05,
    # ---- output ----
    output_name="vlos_projection_broadening.pdf"
):
    """
    Produces a two-panel PDF named 'vlos_projection_broadening.pdf' with a SINGLE continuous colorbar from panel (b).
    """

    # --- compute left panel data (analytic) ---
    sin_i = np.sin(np.radians(incl_deg))
    tan_i = np.tan(np.radians(incl_deg))
    h_func = lambda R: (R * 16.0 + 182.0) / 1000.0  # kpc

    theta_vals = np.linspace(0.0, np.radians(theta_deg_max), int(theta_steps))
    R_vals = np.linspace(R_min, R_max, int(R_steps))
    R_grid, theta_grid = np.meshgrid(R_vals, theta_vals)

    h_grid = h_func(R_grid)
    numerator1 = np.sin(theta_grid) * R_grid + tan_i * h_grid
    numerator2 = np.sin(theta_grid) * R_grid - tan_i * h_grid
    denominator = R_grid * np.cos(theta_grid)
    delta_theta = np.arctan(numerator1 / denominator) - np.arctan(numerator2 / denominator)

    theta_minus = theta_grid - delta_theta / 2.0
    theta_plus = theta_grid + delta_theta / 2.0
    delta_v = v_rot * sin_i * (np.cos(theta_minus) - np.cos(theta_plus))
    delta_v_abs = np.abs(delta_v)

    # --- compute right panel data (observed) ---
    hdu_v = fits.open(vlos_fits_path)[0]
    vlos = hdu_v.data.astype(np.float32)
    wcs = WCS(hdu_v.header)
    hdu_m = fits.open(mom0_fits_path)[0]
    mom0 = hdu_m.data.astype(np.float32)

    if vlos.shape != mom0.shape:
        raise ValueError(f"Shape mismatch: vlos {vlos.shape} vs mom0 {mom0.shape}")

    nhi = mom0_to_nhi_cm2(mom0)

    # mask where invalid or below NHI floor
    mask = ~(np.isfinite(nhi)) | (nhi < float(nhi_mask_min)) | ~(np.isfinite(vlos))
    vlos_masked = np.ma.array(vlos, mask=mask)

    # --- SHARED cmap and continuous norm (no steps) ---
    cmap_shared = plt.get_cmap(cmap_b)
    norm_shared = Normalize(vmin=vmin_b, vmax=vmax_b)

    # --- GridSpec dynamic layout with ONE colorbar on the far right ---
    # columns: [ Panel A | GAP | Panel B | CBAR GAP | CBAR ]
    ratios = [panel_a_width, panel_gap, panel_b_width, cbar_gap_b, cbar_width]
    ratios = np.array(ratios, dtype=float)
    ratios = ratios / np.sum(ratios)
    total = 500
    width_ratios = (ratios * total).astype(int)

    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 120
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(
        1, 5, figure=fig, width_ratios=width_ratios,
        left=0.06, right=0.98, top=0.95, bottom=0.11, wspace=0.0
    )

    # axes
    axA = fig.add_subplot(gs[0, 0])                 # panel (a)
    # gs[0,1] is the gap (no axis)
    axB = fig.add_subplot(gs[0, 2], projection=wcs) # panel (b) with WCS
    # gs[0,3] is the gap between B and cbar
    axCB = fig.add_subplot(gs[0, 4])                # the only colorbar

    # --- draw left (a) using SHARED cmap/norm ---
    imA = axA.pcolormesh(np.degrees(theta_grid), R_grid, delta_v_abs,
                         shading='auto', cmap=cmap_shared, norm=norm_shared)
    axA.set_xlabel(r'$\phi$ [deg]')
    axA.set_ylabel(r'$R$ [kpc]')
    axA.set_xlim([0, theta_deg_max])
    axA.set_ylim([R_min, R_max])
    axA.tick_params(direction='in', length=7, width=1)
    axA.text(0.02, 0.98, "(a)", transform=axA.transAxes,
             ha="left", va="top", fontsize=12, fontweight="bold", color = 'w')

    # --- draw right (b) using the SAME cmap/norm ---
    data_B = np.ma.array(np.abs(vlos_masked) if special_use_abs else vlos_masked)
    imB = axB.imshow(data_B, origin="lower", cmap=cmap_shared, norm=norm_shared, interpolation="nearest")

    # single shared continuous colorbar
    cbar = fig.colorbar(imB, cax=axCB, orientation='vertical')
    cbar.set_label(r"$|\Delta v_{\rm LOS}|$ [km s$^{-1}$]")
    if cbar_ticks is None:
        cbar_ticks = np.linspace(vmin_b, vmax_b, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{int(round(t))}" for t in cbar_ticks])
    axCB.tick_params(direction="in", length=7)

    # N_HI contours on (b)
    if nhi_levels and len(nhi_levels) > 0:
        if contour_white_with_black_edge:
            axB.contour(nhi, levels=np.array(nhi_levels, float), colors="k",
                        linewidths=contour_edge_linewidth, origin="lower")
            axB.contour(nhi, levels=np.array(nhi_levels, float), colors="w",
                        linewidths=contour_linewidth, origin="lower")
        else:
            axB.contour(nhi, levels=np.array(nhi_levels, float), colors="k",
                        linewidths=contour_linewidth, origin="lower")

    # Dashed iso-|Δvlos| lines on (b)
    if sectors:
        h, w = vlos.shape
        if special_center_pix is None:
            special_center_pix = (w / 2.0, h / 2.0)
        theta_map = _angle_map(vlos.shape, special_center_pix, pa_deg=float(special_pa_deg))
        r_map = _radial_map(vlos.shape, special_center_pix)
        base_mask = vlos_masked.mask if special_limit_to_vlos_mask else np.zeros_like(vlos_masked.mask, dtype=bool)
        vlos_for_contour = np.abs(vlos) if special_use_abs else vlos

        clip_mask = np.zeros_like(base_mask)
        if special_clip_to_limits and (xlim is not None or ylim is not None):
            x0, x1 = (0, w - 1) if xlim is None else (int(xlim[0]), int(xlim[1]))
            y0, y1 = (0, h - 1) if ylim is None else (int(ylim[0]), int(ylim[1]))
            clip_mask[:] = True
            clip_mask[y0:y1 + 1, x0:x1 + 1] = False

        for sector in sectors:
            if 'theta_center' in sector and 'half_width' in sector:
                c = float(sector['theta_center']); hw = float(sector['half_width'])
                amin, amax = c - hw, c + hw
            elif 'theta_min' in sector and 'theta_max' in sector:
                amin, amax = float(sector['theta_min']), float(sector['theta_max'])
            else:
                raise ValueError("Each sector must define (theta_center & half_width) or (theta_min & theta_max).")

            r_min = sector.get('r_min', None); r_max = sector.get('r_max', None)
            ang_mask = _in_angle_range(theta_map, amin, amax)
            if r_min is not None: ang_mask &= (r_map >= float(r_min))
            if r_max is not None: ang_mask &= (r_map <= float(r_max))
            region_mask = base_mask | (~ang_mask) | clip_mask
            data_sector = np.ma.array(vlos_for_contour, mask=region_mask)

            for lvl in special_levels:
                if special_edge:
                    axB.contour(data_sector, levels=[float(lvl)], colors=special_edge_color,
                                linewidths=special_edge_width, linestyles=special_linestyle, origin="lower")
                axB.contour(data_sector, levels=[float(lvl)], colors=special_color,
                            linewidths=special_linewidth, linestyles=special_linestyle, origin="lower")

    # Cosmetics for (b)
    axB.set_xlabel("RA (J2000)")
    axB.set_ylabel("Dec (J2000)")
    axB.tick_params(direction='in', which='major', length=7, width=1)
    axB.coords['ra'].set_ticks(spacing=30 * u.arcmin)
    axB.coords['ra'].display_minor_ticks(True)
    axB.coords['ra'].set_minor_frequency(4)
    axB.coords['dec'].set_ticks(spacing=15 * u.arcmin)
    axB.coords['dec'].display_minor_ticks(True)
    axB.coords['dec'].set_minor_frequency(3)
    if xlim is not None: axB.set_xlim(xlim)
    if ylim is not None: axB.set_ylim(ylim)
    axB.text(0.02, 0.98, "(b)", transform=axB.transAxes,
             ha="left", va="top", fontsize=12, fontweight="bold")

    # Save
    plt.savefig(output_name, bbox_inches="tight")
    print(f"Saved: {output_name}")

# ---- example ----
if __name__ == "__main__":
    sectors_example = [
        {'theta_center': 180, 'half_width': 12},
        {'theta_center': 270, 'half_width': 12},
        {'theta_center':   0, 'half_width': 12},
    ]

    plot_ab_panels_shared_cbar_continuous(
        vlos_fits_path="../data/vlos_delta_abs.fits",
        mom0_fits_path="../data/jcomb_submed.mom0.-600-45.fits",
        cmap_b="RdYlBu_r",
        vmin_b=0.0, vmax_b=40.0,
        cbar_ticks=[0, 10, 20, 30, 40],  # optional explicit ticks for a clean continuous bar
        # dashed iso-line settings
        sectors=sectors_example,
        special_levels=(1.7,),
        special_color='w',
        special_linestyle="--",
        special_linewidth=1.0,
        special_edge=True,
        special_edge_color="k",
        special_edge_width=2.0,
        special_pa_deg=38.0,
        # layout knobs
        panel_a_width=0.8,
        panel_gap=0.04,
        panel_b_width=1.0,
        cbar_gap_b=-0.12,
        cbar_width=0.04,
        fig_size=(9.5, 4.8),
        # output fixed:
        output_name="vlos_projection_broadening.pdf"
    )
