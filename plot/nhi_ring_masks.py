#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute H I number densities for individual superbubbles and (optionally) append the results
to a fixed-width catalog, plus a rotated ellipse/ring preview PDF.

Usage
-----
- Edit the `RUN_CONFIG` dictionary near the end of this script to point to the
  desired input bubble table and to set the output filename pattern.
- Execute the script directly: ``python compute_bubble_nhi.py``.

Inputs
------
- Bubble catalog (fixed-width `.tab` file).
- Moment-0 FITS map (`jcomb_vcube_scaled2_mom0.fits` by default).
- Radius FITS map (`r-m31-jcomb_modHeader.fits` by default).
- Rotated moment-0 FITS map for previews
  (`jcomb_submed.mom0.-600-45_rotated.fits` by default), used only for
  plotting panel (a) and (b) preview figures.

Outputs
-------
- Optional fixed-width table containing all original columns plus per-bubble H I densities (off by default).
- Preview PDF (default `nhi_ring_masks.pdf`) showing rotated moment-0 with ellipses/rings colored by $n_{\rm HI,ring}$.
- 默认新增列：`n_HI_center_cm-3`（中心像素）、`n_HI_ellipse_cm-3`（椭圆平均）与 `n_HI_ring_cm-3`（环状区域平均）。
  预览 PDF 由 `plot_mask_preview` / `plot_ring_preview` 控制。

计算流程
--------
1. 根据 `RUN_CONFIG` 读取气泡表，取得每个气泡的像素坐标。
2. 在 moment-0 图上同时提取中心像素值与椭圆区域平均值，并在半径图上采样中心半径 `R`。
3. 使用 `h_pc = 182 + 16 × R_kpc` 和倾角余弦计算视线长度。
4. 将中心与平均柱密度分别除以视线长度得到 `n_HI_center_cm-3`、`n_HI_ellipse_cm-3`。
5. 将掩膜后（bubble 内部置为 NaN）的 moment-0 与放大的椭圆采样，获得环状区域的 `n_HI_ring_cm-3`。
6. 若开启预览选项，则在旋转后的 mom0 背景上绘制 panel (a) 椭圆和 panel (b) 环状区域，
   其中 panel (b) 的环为双层椭圆之间填充颜色的 annulus，并加上 colorbar。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import FuncNorm, PowerNorm, Normalize
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path as MplPath
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -----------------------
# Constants
# -----------------------
PC_TO_CM = 3.0e18
COSI = float(np.cos(np.deg2rad(77.0)))
MOM0_TO_COLUMN = 1.222e6 / 1.42**2 / 3600 * 1.823e18

# For rotated preview to match panel (a) in plot_five_panels_three_rows
ROT_DIRECTION_ANGLE_DEG = -52.0          # vertical up = 0 deg, counterclockwise positive
ROT_ROTATE_BUBBLE_POS_DEG = 52.0         # extra rotation applied to bubble positions
ROT_DELTA_DEG = 52.0                     # PA correction added to angle
ROT_DIST_MPC = 0.761                     # distance to M31
ROT_SCALEBAR_KPC = 5.0                   # length of scalebar in kpc
ROT_A_NORM_KIND = "power"
ROT_A_POWER_GAMMA = 0.6
ROT_A_VMIN = 0.1
ROT_A_VMAX = 20.0
ROT_XLIM_ARCMIN = (-110.0, 110.0)
ROT_YLIM_ARCMIN = (-40.0, 40.0)
USE_TQDM = True


def project_root() -> Path:
    """Return the repository root (parent of the `code` directory)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path_like: str, base_dir: Path) -> Path:
    """
    Resolve a path string relative to the current working directory first and
    fall back to the provided base directory.
    """
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / candidate).resolve()


def _progress(iterable, desc: str):
    """Wrapper for tqdm progress if available/enabled."""
    if USE_TQDM and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def read_fixed_width(path: Path) -> pd.DataFrame:
    """Load a fixed-width bubble catalog and validate essential columns."""
    df = pd.read_fwf(path, header=0, infer_nrows=int(1e6))
    required = {"ra_pix", "dec_pix", "major", "minor", "pa"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def values_at_pixels(data: np.ndarray, x_pix: Iterable[float], y_pix: Iterable[float]) -> np.ndarray:
    """Sample a 2D array at integer pixel locations; out-of-range entries resolve to NaN."""
    data = np.asarray(data)
    x_arr = np.asarray(x_pix, dtype=float)
    y_arr = np.asarray(y_pix, dtype=float)
    xi = np.round(x_arr).astype(int)
    yi = np.round(y_arr).astype(int)
    values = np.full(len(xi), np.nan)
    mask = (
        (xi >= 0) & (yi >= 0) &
        (yi < data.shape[0]) &
        (xi < data.shape[1])
    )
    values[mask] = data[yi[mask], xi[mask]]
    return values


def compute_number_density(column_density: np.ndarray, radius_kpc: np.ndarray) -> np.ndarray:
    """Return n_HI in cm^-3 using the scale height prescription h = 182 + 16 * R[kpc]."""
    h_pc = 182.0 + 16.0 * radius_kpc
    los_length_cm = 2.0 * h_pc * PC_TO_CM / COSI
    return column_density / los_length_cm


def dataframe_to_fixed_width(df: pd.DataFrame, output_path: Path, float_precision: int = 6) -> None:
    """
    Save a dataframe as a left-justified fixed-width text table preserving the original column order.
    """
    def format_float(val: float) -> str:
        if pd.isna(val):
            return "nan"
        return f"{val:.{float_precision}g}"

    str_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_float_dtype(series):
            str_df[col] = series.map(format_float)
        else:
            str_df[col] = series.astype(str)

    widths: Dict[str, int] = {}
    for col in str_df.columns:
        widths[col] = max(len(col), str_df[col].str.len().max())

    header = " ".join(col.ljust(widths[col]) for col in str_df.columns)
    lines = [
        " ".join(str(str_df.iloc[row_idx, col_idx]).ljust(widths[col_name])
                 for col_idx, col_name in enumerate(str_df.columns))
        for row_idx in range(len(str_df))
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        handle.write("\n".join(lines))
        handle.write("\n")


def dated_output_path(code_dir: Path, tag: str = "nhi") -> Path:
    """Construct MMDD-tag.tab (auto-incrementing suffix when collisions occur)."""
    stamp = datetime.now().strftime("%m%d")
    candidate = code_dir / f"{stamp}-{tag}.tab"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        alt = code_dir / f"{stamp}-{tag}_{idx:02d}.tab"
        if not alt.exists():
            return alt
        idx += 1


def ellipse_mean(
    data: np.ndarray,
    x_pix: Iterable[float],
    y_pix: Iterable[float],
    major: Iterable[float],
    minor: Iterable[float],
    pa_deg: Iterable[float],
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return mean values inside each ellipse defined by (major, minor, pa)."""
    data = np.asarray(data, dtype=float)
    height, width = data.shape
    x_arr = np.asarray(x_pix, dtype=float)
    y_arr = np.asarray(y_pix, dtype=float)
    a_arr = np.asarray(major, dtype=float)
    b_arr = np.asarray(minor, dtype=float)
    pa_arr = np.asarray(pa_deg, dtype=float)

    means = np.full(len(x_arr), np.nan, dtype=float)
    union_mask = np.zeros_like(data, dtype=bool) if return_mask else None

    for idx, (x0, y0, a, b, pa) in enumerate(zip(x_arr, y_arr, a_arr, b_arr, pa_arr)):
        if not np.isfinite([x0, y0, a, b, pa]).all():
            continue
        if a <= 0 or b <= 0:
            continue

        span = float(np.hypot(a, b))
        x_min = max(int(np.floor(x0 - span)), 0)
        x_max = min(int(np.ceil(x0 + span)) + 1, width)
        y_min = max(int(np.floor(y0 - span)), 0)
        y_max = min(int(np.ceil(y0 + span)) + 1, height)
        if x_min >= x_max or y_min >= y_max:
            continue

        sub = data[y_min:y_max, x_min:x_max]
        if sub.size == 0:
            continue

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        x_rel = xx - x0
        y_rel = yy - y0

        theta = np.deg2rad(pa)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = x_rel * cos_t + y_rel * sin_t
        y_rot = -x_rel * sin_t + y_rel * cos_t

        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1.0
        if not np.any(mask):
            continue

        values = sub[mask]
        finite = np.isfinite(values)
        if np.any(finite):
            means[idx] = float(np.mean(values[finite]))
            if return_mask:
                union_mask[y_min:y_max, x_min:x_max] |= mask

    if return_mask:
        return means, union_mask
    return means


def _ellipse_bbox(
    x0: float,
    y0: float,
    a: float,
    b: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Return clipped integer bounds [x_min, x_max) and [y_min, y_max) for an ellipse."""
    span = float(np.hypot(a, b))
    x_min = max(int(np.floor(x0 - span)), 0)
    x_max = min(int(np.ceil(x0 + span)) + 1, width)
    y_min = max(int(np.floor(y0 - span)), 0)
    y_max = min(int(np.ceil(y0 + span)) + 1, height)
    return x_min, x_max, y_min, y_max


def _ellipse_mask_on_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    a: float,
    b: float,
    pa_deg: float,
) -> np.ndarray:
    """Return boolean mask of points inside a single ellipse on the provided grid."""
    if not np.isfinite([x0, y0, a, b, pa_deg]).all() or a <= 0 or b <= 0:
        return np.zeros_like(xx, dtype=bool)
    theta = np.deg2rad(pa_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rel = xx - x0
    y_rel = yy - y0
    x_rot = x_rel * cos_t + y_rel * sin_t
    y_rot = -x_rel * sin_t + y_rel * cos_t
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0


def _overlap_stats(
    df: pd.DataFrame,
    shape: tuple[int, int],
    ring_scale: float,
) -> tuple[int, int, int, int, int, int]:
    """
    Count overlaps and areas for bubbles (panel a) and rings (panel b).

    Returns
    -------
    bubble_overlap_count : int
        Number of inner ellipses overlapping at least one other ellipse.
    ring_overlap_count : int
        Number of rings overlapping at least one other ring.
    bubble_overlap_area : int
        Pixels belonging to bubble regions where at least two ellipses overlap.
    bubble_total_area : int
        Pixels covered by any bubble ellipse (union).
    ring_overlap_area : int
        Pixels belonging to ring regions where at least two rings overlap.
    ring_total_area : int
        Pixels covered by any ring (union).
    """
    height, width = shape
    x_pix = df["ra_pix"].to_numpy(float)
    y_pix = df["dec_pix"].to_numpy(float)
    a_pix = df["major"].to_numpy(float)
    b_pix = df["minor"].to_numpy(float)
    pa_deg = df["pa"].to_numpy(float)

    outer_a = a_pix * float(ring_scale)
    outer_b = b_pix * float(ring_scale)

    bubble_counts = np.zeros(shape, dtype=np.uint16)
    ring_counts = np.zeros(shape, dtype=np.uint16)

    # Accumulate per-pixel coverage for bubbles
    for i in _progress(range(len(df)), desc="Accum bubble"):
        x_min, x_max, y_min, y_max = _ellipse_bbox(x_pix[i], y_pix[i], a_pix[i], b_pix[i], width, height)
        if x_min >= x_max or y_min >= y_max:
            continue
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = _ellipse_mask_on_grid(xx, yy, x_pix[i], y_pix[i], a_pix[i], b_pix[i], pa_deg[i])
        if mask.any():
            bubble_counts[y_min:y_max, x_min:x_max][mask] += 1

    # Accumulate per-pixel coverage for rings
    for i in _progress(range(len(df)), desc="Accum ring"):
        xr_min, xr_max, yr_min, yr_max = _ellipse_bbox(x_pix[i], y_pix[i], outer_a[i], outer_b[i], width, height)
        if xr_min >= xr_max or yr_min >= yr_max:
            continue
        yyr, xxr = np.mgrid[yr_min:yr_max, xr_min:xr_max]
        outer_mask = _ellipse_mask_on_grid(xxr, yyr, x_pix[i], y_pix[i], outer_a[i], outer_b[i], pa_deg[i])
        inner_mask = _ellipse_mask_on_grid(xxr, yyr, x_pix[i], y_pix[i], a_pix[i], b_pix[i], pa_deg[i])
        ring_mask = outer_mask & (~inner_mask)
        if ring_mask.any():
            ring_counts[yr_min:yr_max, xr_min:xr_max][ring_mask] += 1

    # Areas
    bubble_total_area = int(np.count_nonzero(bubble_counts >= 1))
    bubble_overlap_area = int(np.count_nonzero(bubble_counts >= 2))
    ring_total_area = int(np.count_nonzero(ring_counts >= 1))
    ring_overlap_area = int(np.count_nonzero(ring_counts >= 2))

    # Overlap counts per object
    bubble_overlaps = np.zeros(len(df), dtype=bool)
    ring_overlaps = np.zeros(len(df), dtype=bool)

    bubble_overlap_mask = bubble_counts >= 2
    ring_overlap_mask = ring_counts >= 2

    for i in _progress(range(len(df)), desc="Mark bubble overlap"):
        x_min, x_max, y_min, y_max = _ellipse_bbox(x_pix[i], y_pix[i], a_pix[i], b_pix[i], width, height)
        if x_min >= x_max or y_min >= y_max:
            continue
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = _ellipse_mask_on_grid(xx, yy, x_pix[i], y_pix[i], a_pix[i], b_pix[i], pa_deg[i])
        if mask.any() and np.any(mask & bubble_overlap_mask[y_min:y_max, x_min:x_max]):
            bubble_overlaps[i] = True

    for i in _progress(range(len(df)), desc="Mark ring overlap"):
        xr_min, xr_max, yr_min, yr_max = _ellipse_bbox(x_pix[i], y_pix[i], outer_a[i], outer_b[i], width, height)
        if xr_min >= xr_max or yr_min >= yr_max:
            continue
        yyr, xxr = np.mgrid[yr_min:yr_max, xr_min:xr_max]
        outer_mask = _ellipse_mask_on_grid(xxr, yyr, x_pix[i], y_pix[i], outer_a[i], outer_b[i], pa_deg[i])
        inner_mask = _ellipse_mask_on_grid(xxr, yyr, x_pix[i], y_pix[i], a_pix[i], b_pix[i], pa_deg[i])
        ring_mask = outer_mask & (~inner_mask)
        if ring_mask.any() and np.any(ring_mask & ring_overlap_mask[yr_min:yr_max, xr_min:xr_max]):
            ring_overlaps[i] = True

    return (
        int(bubble_overlaps.sum()),
        int(ring_overlaps.sum()),
        bubble_overlap_area,
        bubble_total_area,
        ring_overlap_area,
        ring_total_area,
    )


# ============================================================
# Rotated preview helpers (match panel (a) of five_panels script)
# ============================================================

def _read_rot_fits_to_arcmin_with_wcs(fits_path: Path | str):
    """
    Read a rotated 2D FITS and return:
      data, extent_arcmin, header, wcs, cd1_am, cd2_am, nx, ny

    The extent is in arcmin relative to the image center:
      [xmin, xmax, ymin, ymax].
    """
    fits_path = Path(fits_path)
    hdu = fits.open(fits_path)[0]
    data = hdu.data
    hdr = hdu.header
    wcs = WCS(hdr)

    nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
    cd1_am = float(abs(hdr.get("CDELT1", 0.0))) * 60.0  # arcmin per pixel
    cd2_am = float(abs(hdr.get("CDELT2", 0.0))) * 60.0  # arcmin per pixel

    x_center, y_center = nx / 2.0, ny / 2.0
    x_extent = (np.arange(nx) - x_center) * cd1_am
    y_extent = (np.arange(ny) - y_center) * cd2_am
    extent = [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]]
    return data, extent, hdr, wcs, cd1_am, cd2_am, nx, ny


def _rotate_xy(x_am, y_am, deg):
    """
    Rotate arcmin coordinates (x_am, y_am) around the origin counterclockwise by `deg`.
    Supports scalars or arrays.
    """
    theta = np.radians(deg)
    ct, st = np.cos(theta), np.sin(theta)
    xr = x_am * ct - y_am * st
    yr = x_am * st + y_am * ct
    return xr, yr


def _ellipse_ring_path(
    xc: float,
    yc: float,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    angle_deg: float,
    n_vertices: int = 100,
) -> MplPath:
    """
    Construct a Path for an elliptical annulus (outer ellipse minus inner ellipse).

    Parameters
    ----------
    xc, yc : center in data coordinates.
    a_in, b_in : inner semi axes in data units (arcmin).
    a_out, b_out : outer semi axes in data units (arcmin).
    angle_deg : rotation angle of ellipse (deg, counterclockwise from x axis).
    n_vertices : sampling resolution.

    Returns
    -------
    path : matplotlib.path.Path representing the ring.
    """
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)

    t = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)

    # Outer ellipse, traversed counterclockwise
    xo = a_out * np.cos(t)
    yo = b_out * np.sin(t)
    xo_r = xc + xo * ct - yo * st
    yo_r = yc + xo * st + yo * ct

    # Inner ellipse, traversed clockwise (reverse order)
    ti = t[::-1]
    xi = a_in * np.cos(ti)
    yi = b_in * np.sin(ti)
    xi_r = xc + xi * ct - yi * st
    yi_r = yc + xi * st + yi * ct

    vertices = np.concatenate(
        [
            np.column_stack([xo_r, yo_r]),
            np.column_stack([xi_r, yi_r]),
            np.array([[xo_r[0], yo_r[0]]]),
        ],
        axis=0,
    )

    codes = np.full(len(vertices), MplPath.LINETO, dtype=MplPath.code_type)
    codes[0] = MplPath.MOVETO
    codes[len(xo_r)] = MplPath.MOVETO  # start inner loop
    codes[-1] = MplPath.CLOSEPOLY

    return MplPath(vertices, codes)


def plot_rotated_mask_two_panels(
    rot_mom0_fits: Path | str,
    df: pd.DataFrame,
    ring_scale: float,
    output_path: Path | str,
    show_inner: bool = True,
    show_ring: bool = True,
) -> None:
    """
    Plot a two-row figure on the rotated mom0 background that matches panel (a)
    in the five_panels script.

    Panel (a): inner ellipses (bubble body).
    Panel (b): color filled elliptical ring region, composed of two ellipses
               (inner original, outer scaled by `ring_scale`) with the area
               in between filled with color.

    Color coding:
      Both panels use color mapped from `n_HI_ring_cm-3` with a single colorbar
      appended to panel (b). The colorbar describes the ellipse colors, not the
      grayscale background.

    The function expects the bubble table `df` to contain:
      - ra_hms, dec_dms (for SkyCoord)
      - maj_as, min_as (arcsec)
      - pa (deg)
      - n_HI_ring_cm-3
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load rotated map
    data_a, extent_a, hdr_a, wcs_a, cd1_am, cd2_am, nx_a, ny_a = _read_rot_fits_to_arcmin_with_wcs(rot_mom0_fits)

    # Background normalization
    if str(ROT_A_NORM_KIND).lower() == "power":
        norm_a = PowerNorm(gamma=ROT_A_POWER_GAMMA, vmin=ROT_A_VMIN, vmax=ROT_A_VMAX)
    else:
        norm_a = FuncNorm((np.arcsinh, np.sinh), vmin=ROT_A_VMIN, vmax=ROT_A_VMAX)

    # Color for n_HI_ring
    if "n_HI_ring_cm-3" in df.columns:
        n_ring = df["n_HI_ring_cm-3"].to_numpy(float)
        finite_n = np.isfinite(n_ring)
        if np.any(finite_n):
            vmin_ring = float(np.nanpercentile(n_ring[finite_n], 5.0))
            vmax_ring = float(np.nanpercentile(n_ring[finite_n], 95.0))
        else:
            vmin_ring, vmax_ring = 0.05, 0.40
    else:
        n_ring = np.full(len(df), np.nan)
        vmin_ring, vmax_ring = 0.05, 0.40

    norm_n = Normalize(vmin=vmin_ring, vmax=vmax_ring)
    cmap_n = plt.get_cmap("coolwarm")

    # Coordinates of bubbles on rotated map
    if not all(k in df.columns for k in ("ra_hms", "dec_dms", "maj_as", "min_as", "pa")):
        raise KeyError(
            "Bubble table must contain ra_hms, dec_dms, maj_as, min_as, pa "
            "for rotated preview plotting."
        )

    ra = df["ra_hms"].astype(str).values
    dec = df["dec_dms"].astype(str).values
    coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    x_pix, y_pix = wcs_a.world_to_pixel(coords)
    x_am = (x_pix - nx_a / 2.0) * cd1_am
    y_am = (y_pix - ny_a / 2.0) * cd2_am
    x_am_rot, y_am_rot = _rotate_xy(x_am, y_am, ROT_ROTATE_BUBBLE_POS_DEG)

    maj_as = df["maj_as"].to_numpy(float)
    min_as = df["min_as"].to_numpy(float)
    pa_deg = df["pa"].to_numpy(float)

    # Prepare figure: two rows, shared x-axis
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.5), sharex=True)
    ax_a, ax_b = axes

    # Helper to draw background, ticks, scalebar, N/W arrow and label
    def _setup_background(ax, panel_label: str):
        ax.imshow(data_a, origin="lower", cmap="gray", norm=norm_a, extent=extent_a)
        ax.set_xlim(ROT_XLIM_ARCMIN)
        ax.set_ylim(ROT_YLIM_ARCMIN)
        ax.tick_params(axis="both", color="white", direction="in")

        # Tick density
        ax.xaxis.set_major_locator(MultipleLocator(50.0))
        ax.yaxis.set_major_locator(MultipleLocator(20.0))

        ax.tick_params(labelbottom=True, labelleft=True, labelcolor="black")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}'"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}'"))

        # Panel label
        ax.text(
            0.02,
            0.97,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            va="top",
        )

        # Scalebar
        x_min, x_max = ROT_XLIM_ARCMIN
        y_min, y_max = ROT_YLIM_ARCMIN
        arcsec_per_rad = 206265.0
        theta_as = ROT_SCALEBAR_KPC * 1e3 / (ROT_DIST_MPC * 1e6) * arcsec_per_rad
        scalebar_arcmin = theta_as / 60.0
        x0 = x_min + 0.90 * (x_max - x_min)
        y0 = y_min + 0.10 * (y_max - y_min)
        ax.plot(
            [x0 - scalebar_arcmin / 2.0, x0 + scalebar_arcmin / 2.0],
            [y0, y0],
            color="white",
            lw=1.0,
        )
        ax.text(
            x0,
            y0 + 8.1,
            f"{ROT_SCALEBAR_KPC:.0f} kpc",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
        )

        # N/W arrow (same as five_panels panel a)
        arrow_base_x = x_min + 0.11 * (x_max - x_min)
        arrow_base_y = y_min + 0.08 * (y_max - y_min)
        arrow_len = 0.04 * (x_max - x_min)
        theta_dir = np.radians(ROT_DIRECTION_ANGLE_DEG)

        dx_N = arrow_len * np.sin(theta_dir)
        dy_N = arrow_len * np.cos(theta_dir)
        ax.arrow(
            arrow_base_x,
            arrow_base_y,
            dx_N,
            dy_N,
            head_width=1,
            head_length=1,
            fc="white",
            ec="white",
            linewidth=0.7,
        )
        ax.text(
            arrow_base_x + dx_N * 1.70,
            arrow_base_y + dy_N * 1.70,
            "N",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
        )

        theta_W = theta_dir + np.pi / 2.0
        dx_W = arrow_len * np.sin(theta_W)
        dy_W = arrow_len * np.cos(theta_W)
        ax.arrow(
            arrow_base_x,
            arrow_base_y,
            dx_W,
            dy_W,
            head_width=1,
            head_length=1,
            fc="white",
            ec="white",
            linewidth=0.7,
        )
        ax.text(
            arrow_base_x + dx_W * 1.70,
            arrow_base_y + dy_W * 1.70,
            "W",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
        )

    # Panel (a): inner ellipses
    _setup_background(ax_a, "(a)")
    if show_inner:
        for i in range(len(df)):
            angle_plot = float(pa_deg[i]) + 90.0 + ROT_DELTA_DEG
            val_ring = n_ring[i]
            color_rgba = cmap_n(norm_n(val_ring)) if np.isfinite(val_ring) else (0.5, 0.5, 0.5, 0.5)
            face = color_rgba[:3] + (0.65,)
            edge = color_rgba[:3] + (0.0,)
            ax_a.add_patch(
                patches.Ellipse(
                    (x_am_rot[i], y_am_rot[i]),
                    width=(min_as[i] / 60.0) * 2.0,
                    height=(maj_as[i] / 60.0) * 2.0,
                    angle=angle_plot,
                    facecolor=face,
                    edgecolor=edge,
                    lw=0.7,
                    zorder=6,
                )
            )

    # Panel (b): filled ring annulus
    _setup_background(ax_b, "(b)")
    if show_ring:
        for i in range(len(df)):
            val_ring = n_ring[i]
            color_rgba = cmap_n(norm_n(val_ring)) if np.isfinite(val_ring) else (0.5, 0.5, 0.5, 0.5)

            # Inner and outer semi-axes in arcmin
            a_in = (min_as[i] / 60.0)
            b_in = (maj_as[i] / 60.0)
            a_out = a_in * float(ring_scale)
            b_out = b_in * float(ring_scale)

            angle_plot = float(pa_deg[i]) + 90.0 + ROT_DELTA_DEG

            ring_path = _ellipse_ring_path(
                xc=float(x_am_rot[i]),
                yc=float(y_am_rot[i]),
                a_in=a_in,
                b_in=b_in,
                a_out=a_out,
                b_out=b_out,
                angle_deg=angle_plot,
                n_vertices=160,
            )

            # 半透明风格与 panel (a) 一致
            face_rgba = color_rgba[:3] + (0.65,)
            edge_rgba = color_rgba[:3] + (0.0,)

            ax_b.add_patch(
                patches.PathPatch(
                    ring_path,
                    facecolor=face_rgba,
                    edgecolor=edge_rgba,
                    lw=0.7,
                    zorder=6,
                )
            )

    ax_b.set_xlabel("arcmin")
    ax_b.set_ylabel("arcmin")

    # 先对两个 panel 做 tight_layout，预留右侧空间
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

    # Colorbar 悬挂在 panel (b) 右侧
    bbox_b = ax_b.get_position()
    cax_width = 0.02
    cax_pad = 0.01
    cax_left = bbox_b.x1 + cax_pad
    cax = fig.add_axes([cax_left, bbox_b.y0, cax_width, bbox_b.height])

    sm = plt.cm.ScalarMappable(cmap=cmap_n, norm=norm_n)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r"$n_{\rm HI}$ [cm$^{-3}$]")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    bubble_tab: Path,
    output_tab: Path | None,
    mom0_fits: Path,
    r_fits: Path,
    overwrite: bool = False,
    plot_mask_preview: bool = False,
    mask_preview_pdf: Path | None = None,
    ring_scale: float = 2.0,
    plot_ring_preview: bool = False,
    ring_preview_pdf: Path | None = None,
    rot_mom0_fits: Path | None = None,
    output_pdf: Path | None = None,
) -> Path | None:
    """Core implementation that reads inputs, computes n_HI, and optionally writes outputs."""
    if (output_tab is not None) and output_tab.exists() and not overwrite:
        raise FileExistsError(f"{output_tab} already exists. Use overwrite=True to replace it.")

    df = read_fixed_width(bubble_tab)
    print(f"[nHI] Loaded {len(df)} rows from {bubble_tab}")

    mom0_map = fits.getdata(mom0_fits)
    r_map = fits.getdata(r_fits)

    mom0_center = values_at_pixels(mom0_map, df["ra_pix"], df["dec_pix"])
    mom0_ellipse, ellipse_mask = ellipse_mean(
        mom0_map,
        df["ra_pix"],
        df["dec_pix"],
        df["major"],
        df["minor"],
        df["pa"],
        return_mask=True,
    )
    radius_pc = values_at_pixels(r_map, df["ra_pix"], df["dec_pix"])
    radius_kpc = radius_pc / 1e3

    (
        bubble_overlap_count,
        ring_overlap_count,
        bubble_overlap_area,
        bubble_total_area,
        ring_overlap_area,
        ring_total_area,
    ) = _overlap_stats(
        df=df,
        shape=mom0_map.shape,
        ring_scale=ring_scale,
    )
    print(
        f"[nHI] Panel (a): {bubble_overlap_count}/{len(df)} bubbles overlap with at least one other bubble."
    )
    print(
        f"[nHI] Panel (b): {ring_overlap_count}/{len(df)} rings overlap with at least one other ring."
    )
    if bubble_total_area > 0:
        frac_bubble_overlap = bubble_overlap_area / bubble_total_area
        print(
            f"[nHI] Panel (a) overlap area: {bubble_overlap_area} / {bubble_total_area} pixels "
            f"({frac_bubble_overlap:.3f})"
        )
    if ring_total_area > 0:
        frac_ring_overlap = ring_overlap_area / ring_total_area
        print(
            f"[nHI] Panel (b) overlap area: {ring_overlap_area} / {ring_total_area} pixels "
            f"({frac_ring_overlap:.3f})"
        )

    masked_map = np.array(mom0_map, dtype=float, copy=True)
    masked_map[ellipse_mask] = np.nan

    expanded_major_pix = np.asarray(df["major"], dtype=float) * float(ring_scale)
    expanded_minor_pix = np.asarray(df["minor"], dtype=float) * float(ring_scale)

    if plot_ring_preview:
        mom0_ring = ellipse_mean(
            masked_map,
            df["ra_pix"],
            df["dec_pix"],
            expanded_major_pix,
            expanded_minor_pix,
            df["pa"],
        )
    else:
        mom0_ring = ellipse_mean(
            masked_map,
            df["ra_pix"],
            df["dec_pix"],
            expanded_major_pix,
            expanded_minor_pix,
            df["pa"],
        )

    column_density_center = mom0_center * MOM0_TO_COLUMN
    column_density_ellipse = mom0_ellipse * MOM0_TO_COLUMN
    column_density_ring = mom0_ring * MOM0_TO_COLUMN
    n_hi_center = compute_number_density(column_density_center, radius_kpc)
    n_hi_ellipse = compute_number_density(column_density_ellipse, radius_kpc)
    n_hi_ring = compute_number_density(column_density_ring, radius_kpc)

    df_out = df.copy()
    for col in ("n_HI_cm-3", "n_HI_center_cm-3", "n_HI_ellipse_cm-3", "n_HI_ring_cm-3"):
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)
    df_out.insert(len(df_out.columns), "n_HI_center_cm-3", n_hi_center)
    df_out.insert(len(df_out.columns), "n_HI_ellipse_cm-3", n_hi_ellipse)
    df_out.insert(len(df_out.columns), "n_HI_ring_cm-3", n_hi_ring)

    finite_center = np.isfinite(n_hi_center)
    finite_ellipse = np.isfinite(n_hi_ellipse)
    finite_ring = np.isfinite(n_hi_ring)

    if finite_center.any():
        print(
            "[nHI] n_HI_center range: "
            f"{n_hi_center[finite_center].min():.3g} - "
            f"{n_hi_center[finite_center].max():.3g} cm^-3"
        )
    else:
            print("[nHI] Warning: no finite center n_HI values computed.")
    if finite_ellipse.any():
        print(
            "[nHI] n_HI_ellipse range: "
            f"{n_hi_ellipse[finite_ellipse].min():.3g} - "
            f"{n_hi_ellipse[finite_ellipse].max():.3g} cm^-3"
        )
    else:
        print("[nHI] Warning: no finite ellipse-averaged n_HI values computed.")
    if finite_ring.any():
        print(
            "[nHI] n_HI_ring range: "
            f"{n_hi_ring[finite_ring].min():.3g} - "
            f"{n_hi_ring[finite_ring].max():.3g} cm^-3"
        )
    else:
        print("[nHI] Warning: no finite ring-averaged n_HI values computed.")

    if output_tab is not None:
        dataframe_to_fixed_width(df_out, output_tab)
        print(f"[nHI] Augmented table written to {output_tab}")

    # Combined rotated preview: panel (a) inner ellipses, panel (b) ring ellipses
    if plot_mask_preview or plot_ring_preview:
        # Choose output path: prefer explicitly given, otherwise derive from output_tab
        if mask_preview_pdf is not None:
            preview_path = mask_preview_pdf
        elif ring_preview_pdf is not None:
            preview_path = ring_preview_pdf
        elif output_pdf is not None:
            preview_path = output_pdf
        elif output_tab is not None:
            preview_path = output_tab.with_name(output_tab.stem + "_rotmask.pdf")
        else:
            preview_path = None

        # Use rotated mom0 fits if provided, otherwise fall back to mom0_fits
        rot_fits_path = rot_mom0_fits if rot_mom0_fits is not None else mom0_fits

        # Use df_out so that n_HI_ring_cm-3 is available for color mapping
        if preview_path is not None:
            try:
                plot_rotated_mask_two_panels(
                    rot_mom0_fits=rot_fits_path,
                    df=df_out,
                    ring_scale=ring_scale,
                    output_path=preview_path,
                    show_inner=plot_mask_preview,
                    show_ring=plot_ring_preview,
                )
                print(f"[nHI] Rotated ellipse and ring mask preview saved to {preview_path}")
            except Exception as exc:
                print(f"[nHI] Warning: failed to generate rotated mask preview: {exc}")
        else:
            print("[nHI] Skipping preview: no output path provided.")

    return output_tab


def main(
    bubble_tab: str | Path,
    output_tab: str | Path | None = None,
    mom0_fits: str | Path | None = None,
    r_fits: str | Path | None = None,
    overwrite: bool = False,
    plot_mask_preview: bool = False,
    mask_preview_pdf: str | Path | None = None,
    ring_scale: float = 2.0,
    plot_ring_preview: bool = False,
    ring_preview_pdf: str | Path | None = None,
    rot_mom0_fits: str | Path | None = None,
    output_pdf: str | Path | None = None,
) -> Path | None:
    """
    Run the n_HI augmentation workflow and return the path to the written table.
    Parameters may be omitted to fall back to default project data products.
    """
    base = project_root()
    code_dir = base / "code"
    plot_dir = base / "plot"

    mom0_default = base / "data" / "jcomb_vcube_scaled2_mom0.fits"
    r_default = base / "data" / "r-m31-jcomb_modHeader.fits"
    rot_mom0_default = base / "data" / "jcomb_submed.mom0.-600-45_rotated.fits"

    bubble_tab_path = resolve_path(str(bubble_tab), base)

    output_tab_path = None if output_tab is None else resolve_path(str(output_tab), base)

    mom0_path = resolve_path(str(mom0_fits or mom0_default), base)
    r_path = resolve_path(str(r_fits or r_default), base)

    if rot_mom0_fits is None:
        rot_mom0_path = rot_mom0_default
    else:
        rot_mom0_path = resolve_path(str(rot_mom0_fits), base)

    preview_path = None if mask_preview_pdf is None else resolve_path(str(mask_preview_pdf), base)
    ring_preview_path = None if ring_preview_pdf is None else resolve_path(str(ring_preview_pdf), base)
    output_pdf_path = resolve_path(str(output_pdf), base) if output_pdf is not None else (plot_dir / "nhi_ring_masks.pdf")

    return run_pipeline(
        bubble_tab=bubble_tab_path,
        output_tab=output_tab_path,
        mom0_fits=mom0_path,
        r_fits=r_path,
        overwrite=overwrite,
        plot_mask_preview=plot_mask_preview,
        mask_preview_pdf=preview_path,
        ring_scale=ring_scale,
        plot_ring_preview=plot_ring_preview,
        ring_preview_pdf=ring_preview_path,
        rot_mom0_fits=rot_mom0_path,
        output_pdf=output_pdf_path,
    )


if __name__ == "__main__":
    PROJECT_BASE = project_root()
    CODE_DIR = PROJECT_BASE / "code"
    DATA_DIR = PROJECT_BASE / "data"
    PLOT_DIR = PROJECT_BASE / "plot"

    # === Adjust these parameters as needed ===
    RUN_CONFIG = {
        "bubble_tab": CODE_DIR / "1113.tab",
        "output_tab": None,  # set to a path to write augmented table; None disables table output
        "mom0_fits": DATA_DIR / "jcomb_vcube_scaled2_mom0.fits",
        "r_fits": DATA_DIR / "r-m31-jcomb_modHeader.fits",
        # Rotated mom0 fits used for previews (panel a/b)
        "rot_mom0_fits": DATA_DIR / "jcomb_submed.mom0.-600-45_rotated.fits",
        "overwrite": True,
        "plot_mask_preview": True,
        "mask_preview_pdf": None,
        "ring_scale": 2.0,
        "plot_ring_preview": True,
        "ring_preview_pdf": None,
        "output_pdf": PLOT_DIR / "nhi_ring_masks.pdf",
    }

    RESULT_PATH = main(**RUN_CONFIG)
    print(f"[nHI] Finished: {RESULT_PATH}")
