#!/usr/bin/env python3
"""Plot a full-disk mom0 map with cone overlays and six pre-ROHSA spectra."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = REPO_ROOT / ".local"
MPL_DIR = LOCAL_DIR / "matplotlib"
CACHE_DIR = LOCAL_DIR / "cache"
ASTROPY_CACHE_DIR = CACHE_DIR / "astropy"
for directory in (MPL_DIR, CACHE_DIR, ASTROPY_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("ASTROPY_CACHE_DIR", str(ASTROPY_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils import iers
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys

sys.path.insert(0, str(REPO_ROOT / "code"))
from bubble_catalog import read_catalog


DEFAULT_MOM0 = (
    REPO_ROOT
    / "data"
    / "jcomb_submed.mom0.-600-45.fits"
)
DEFAULT_CUBE = REPO_ROOT / "data" / "hi_cube_jcomb_scaled2_submed.fits"
DEFAULT_MOM1 = REPO_ROOT / "data" / "jcomb_vcube_scaled2_submed.fits.moment.weighted_coord"
DEFAULT_MOM2 = REPO_ROOT / "data" / "jcomb_vcube_scaled2_submed.fits.moment.weighted_dispersion_coord"
DEFAULT_R = REPO_ROOT / "data" / "r-m31-jcomb_modHeader.fits"
DEFAULT_ALPHA = REPO_ROOT / "data" / "alpha-m31-jcomb_modHeader.fits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "rohsa"
DEFAULT_MIDLINE_CACHE_DIR = REPO_ROOT / "data" / "rohsa_spectra_cache"
DEFAULT_MASKS = {
    "maj": REPO_ROOT / "data" / "cone_masks" / "m31_cone_maj_mask.fits",
    "miE": REPO_ROOT / "data" / "cone_masks" / "m31_cone_miE_mask.fits",
    "miW": REPO_ROOT / "data" / "cone_masks" / "m31_cone_miW_mask.fits",
}
DEFAULT_ROHSA_DIRS = {
    "maj": REPO_ROOT / "data" / "cone_cutouts" / "rohsa" / "maj_cutout_ng3" / "summary_ng3",
    "miE": REPO_ROOT / "data" / "cone_cutouts" / "rohsa" / "miE_cutout_ng3" / "summary_ng3",
    "miW": REPO_ROOT / "data" / "cone_cutouts" / "rohsa" / "miW_cutout_ng3" / "summary_ng3",
}
DISPLAY_R_KPC_RANGE_BY_SECTOR = {
    "maj": (3.0, 25.0),
    "miE": (3.0, 33.0),
    "miW": (3.0, 33.0),
}
SECTOR_CONFIG = {
    "maj": {
        "axis_alpha_deg": 180.0,
        "marker": "o",
        "target_radii_kpc": (10.0, 15.0),
        "cutout_path": REPO_ROOT / "data" / "cone_cutouts" / "hi_cube_jcomb_scaled2_submed_cone_maj_cutout.fits",
    },
    "miE": {
        "axis_alpha_deg": 90.0,
        "marker": "s",
        "target_radii_kpc": (10.0, 19.0),
        "cutout_path": REPO_ROOT / "data" / "cone_cutouts" / "hi_cube_jcomb_scaled2_submed_cone_miE_cutout.fits",
    },
    "miW": {
        "axis_alpha_deg": 270.0,
        "marker": "D",
        "target_radii_kpc": (12.0, 20.0),
        "cutout_path": REPO_ROOT / "data" / "cone_cutouts" / "hi_cube_jcomb_scaled2_submed_cone_miW_cutout.fits",
    },
}
SPECTRA_CACHE_INDEX = "sampled_spectra_index.json"
PANEL_LETTERS = ("b", "c", "d", "e", "f", "g")
SPECTRUM_COLOR = "#333333"
ROHSA_TOTAL_COLOR = "#c23b22"
ROHSA_COMPONENT_COLOR = "#c23b22"
ROHSA_WINDOW_SCALE = 1.62


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mom0", default=str(DEFAULT_MOM0))
    parser.add_argument("--cube", default=str(DEFAULT_CUBE))
    parser.add_argument("--mom1", default=str(DEFAULT_MOM1))
    parser.add_argument("--mom2", default=str(DEFAULT_MOM2))
    parser.add_argument("--r-map", default=str(DEFAULT_R))
    parser.add_argument("--alpha-map", default=str(DEFAULT_ALPHA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-basename", default="pre_rohsa_cone_spectra_overview")
    parser.add_argument("--midline-cache-dir", default=str(DEFAULT_MIDLINE_CACHE_DIR))
    parser.add_argument("--xlim", nargs=2, type=float, default=(450.0, 2550.0))
    parser.add_argument("--ylim", nargs=2, type=float, default=(300.0, 2700.0))
    parser.add_argument("--vmin-nhi", type=float, default=0.0)
    parser.add_argument("--vmax-nhi", type=float, default=40.0)
    parser.add_argument("--radius-tolerance-pc", type=float, default=1500.0)
    parser.add_argument("--alpha-weight-pc-per-deg", type=float, default=250.0)
    parser.add_argument("--velocity-min-halfwidth", type=float, default=22.0)
    parser.add_argument("--velocity-max-halfwidth", type=float, default=140.0)
    parser.add_argument("--velocity-sigma-scale", type=float, default=3.5)
    parser.add_argument("--rohsa-ng", type=int, default=3)
    parser.add_argument("--highlight-catalog", default=None)
    parser.add_argument("--highlight-id", type=int, default=None)
    parser.add_argument("--refresh-spectra", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_float_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    return data, header


def moment_to_nhi(moment_data: np.ndarray, header: fits.Header) -> np.ndarray:
    del header
    return np.asarray(moment_data, dtype=np.float32) * np.float32(1.222e6 / (1.4**2) / 60.0 / 60.0 * 1.8e18 / 1.0e20)


def spectral_axis_kms(header: fits.Header, nchan: int) -> np.ndarray:
    try:
        axis = WCS(header).spectral.pixel_to_world(np.arange(nchan, dtype=float))
        return np.asarray(axis.to(u.km / u.s).value, dtype=np.float32)
    except Exception:
        crpix = float(header["CRPIX3"])
        crval = float(header["CRVAL3"])
        cdelt = float(header["CDELT3"])
        unit = u.Unit(header.get("CUNIT3", "m/s"))
        values = (np.arange(nchan, dtype=float) + 1.0 - crpix) * cdelt + crval
        return np.asarray((values * unit).to(u.km / u.s).value, dtype=np.float32)


def circular_angle_distance_deg(values: np.ndarray, target: float) -> np.ndarray:
    delta = (values - target + 180.0) % 360.0 - 180.0
    return np.abs(delta)


def choose_sampling_points(
    r_pc: np.ndarray,
    alpha_deg: np.ndarray,
    mask_map: dict[str, np.ndarray],
    radius_tolerance_pc: float,
    alpha_weight_pc_per_deg: float,
) -> list[dict[str, float | int | str]]:
    points: list[dict[str, float | int | str]] = []
    occupied = np.zeros_like(r_pc, dtype=bool)

    for sector_name, sector_info in SECTOR_CONFIG.items():
        sector_mask = np.asarray(mask_map[sector_name], dtype=bool)
        base_valid = sector_mask & np.isfinite(r_pc) & np.isfinite(alpha_deg)

        for radius_kpc in sector_info["target_radii_kpc"]:
            target_pc = radius_kpc * 1000.0
            score = np.abs(r_pc - target_pc) + alpha_weight_pc_per_deg * circular_angle_distance_deg(
                alpha_deg, float(sector_info["axis_alpha_deg"])
            )
            valid = base_valid & (np.abs(r_pc - target_pc) <= radius_tolerance_pc) & (~occupied)
            if not np.any(valid):
                valid = base_valid & (~occupied)
            score = np.where(valid, score, np.inf)
            y, x = np.unravel_index(np.argmin(score), score.shape)
            occupied[y, x] = True
            points.append(
                {
                    "sector": sector_name,
                    "radius_kpc": float(radius_kpc),
                    "target_radius_pc": float(target_pc),
                    "x": int(x),
                    "y": int(y),
                    "r_pc": float(r_pc[y, x]),
                    "alpha_deg": float(alpha_deg[y, x]),
                    "axis_alpha_deg": float(sector_info["axis_alpha_deg"]),
                    "score": float(score[y, x]),
                }
            )
    return points


def spectrum_cache_paths(output_dir: Path, item: dict[str, float | int | str]) -> tuple[Path, str]:
    stem = f"{item['sector']}_r{int(round(float(item['radius_kpc']))):02d}kpc_x{int(item['x']):04d}_y{int(item['y']):04d}"
    return output_dir / "sampled_spectra" / f"{stem}.txt", stem


def cache_matches(index_payload: dict, cube_path: Path, points: list[dict[str, float | int | str]]) -> bool:
    if index_payload.get("cube_path") != str(cube_path):
        return False
    cached = index_payload.get("spectra", [])
    if len(cached) != len(points):
        return False
    for current, old in zip(points, cached):
        if str(old.get("sector")) != str(current.get("sector")):
            return False
        if int(old.get("x")) != int(current.get("x")) or int(old.get("y")) != int(current.get("y")):
            return False
        if int(round(float(old.get("radius_kpc")))) != int(round(float(current.get("radius_kpc")))):
            return False
    return True


def load_cached_spectra_from_index(index_path: Path, points: list[dict[str, float | int | str]]) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    if not index_path.exists():
        return None
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    text_paths = {(
        str(item["sector"]),
        int(round(float(item["radius_kpc"]))),
        int(item["x"]),
        int(item["y"]),
    ): Path(item["text_path"]) for item in index_payload.get("spectra", [])}
    spectra: list[dict[str, object]] = []
    velocity_ref: np.ndarray | None = None
    for item in points:
        key = (str(item["sector"]), int(round(float(item["radius_kpc"]))), int(item["x"]), int(item["y"]))
        txt_path = text_paths.get(key)
        if not txt_path.exists():
            return None
        table = np.loadtxt(txt_path, comments="#", dtype=float)
        if table.ndim != 2 or table.shape[1] < 2:
            return None
        velocity = np.asarray(table[:, 0], dtype=np.float32)
        spectrum = np.asarray(table[:, 1], dtype=np.float32)
        stem = txt_path.stem
        if velocity_ref is None:
            velocity_ref = velocity
        spectra.append(
            {
                **item,
                "velocity_kms": velocity,
                "spectrum": spectrum,
                "finite_channel_count": int(np.isfinite(spectrum).sum()),
                "cache_stem": stem,
            }
        )
    if velocity_ref is None:
        return None
    return velocity_ref, spectra


def load_cached_spectra(
    output_dir: Path,
    cube_path: Path,
    points: list[dict[str, float | int | str]],
    *,
    fallback_cache_dir: Path | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    primary_index = output_dir / SPECTRA_CACHE_INDEX
    if primary_index.exists():
        index_payload = json.loads(primary_index.read_text(encoding="utf-8"))
        if cache_matches(index_payload, cube_path, points):
            cached = load_cached_spectra_from_index(primary_index, points)
            if cached is not None:
                return cached
    if fallback_cache_dir is not None:
        fallback_index = fallback_cache_dir / "cone_midline_spectra_index.json"
        if fallback_index.exists():
            index_payload = json.loads(fallback_index.read_text(encoding="utf-8"))
            if index_payload.get("cube_path") == str(cube_path):
                cached = load_cached_spectra_from_index(fallback_index, points)
                if cached is not None:
                    return cached
    return None


def write_spectrum_text(output_dir: Path, item: dict[str, object]) -> str:
    txt_path, stem = spectrum_cache_paths(output_dir, item)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    velocity = np.asarray(item["velocity_kms"], dtype=float)
    spectrum = np.asarray(item["spectrum"], dtype=float)
    header = [
        f"# sector={item['sector']}",
        f"# radius_kpc={float(item['radius_kpc']):.2f}",
        f"# x={int(item['x'])}",
        f"# y={int(item['y'])}",
        f"# r_pc={float(item['r_pc']):.3f}",
        f"# alpha_deg={float(item['alpha_deg']):.3f}",
        "# columns: velocity_kms intensity_jy_per_beam",
    ]
    data = np.column_stack([velocity, spectrum])
    np.savetxt(txt_path, data, fmt="%.10e", header="\n".join(header), comments="")
    return stem


def write_spectra_cache_index(output_dir: Path, cube_path: Path, spectra: list[dict[str, object]]) -> None:
    index_path = output_dir / SPECTRA_CACHE_INDEX
    payload = {
        "cube_path": str(cube_path),
        "spectra": [
            {
                "sector": str(item["sector"]),
                "radius_kpc": float(item["radius_kpc"]),
                "x": int(item["x"]),
                "y": int(item["y"]),
                "text_path": str((output_dir / "sampled_spectra" / f"{item['cache_stem']}.txt").resolve()),
            }
            for item in spectra
        ],
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_spectrum(
    cube_path: Path,
    points: list[dict[str, float | int | str]],
    output_dir: Path,
    *,
    fallback_cache_dir: Path | None = None,
    refresh: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]], bool]:
    if not refresh:
        cached = load_cached_spectra(output_dir, cube_path, points, fallback_cache_dir=fallback_cache_dir)
        if cached is not None:
            velocities, spectra = cached
            return velocities, spectra, False

    with fits.open(cube_path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()

    while cube.ndim > 3:
        cube = cube[0]
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    velocities = spectral_axis_kms(header, cube.shape[0])
    spectra: list[dict[str, object]] = []
    for point in points:
        x = int(point["x"])
        y = int(point["y"])
        spectrum = np.asarray(cube[:, y, x], dtype=np.float32)
        item = {
            **point,
            "velocity_kms": velocities,
            "spectrum": spectrum,
            "finite_channel_count": int(np.isfinite(spectrum).sum()),
        }
        item["cache_stem"] = write_spectrum_text(output_dir, item)
        spectra.append(
            {
                **item,
            }
        )
    write_spectra_cache_index(output_dir, cube_path, spectra)
    return velocities, spectra, True


def load_rohsa_models(
    full_header: fits.Header,
    spectra: list[dict[str, object]],
) -> None:
    full_wcs = WCS(full_header).celestial

    for item in spectra:
        sector = str(item["sector"])
        summary_dir = DEFAULT_ROHSA_DIRS[sector]
        model_cube_path = summary_dir / "model_cube.fits"
        cutout_path = Path(SECTOR_CONFIG[sector]["cutout_path"])
        if not model_cube_path.exists():
            item["rohsa_velocity_kms"] = np.array([], dtype=np.float32)
            item["rohsa_model_total"] = np.array([], dtype=np.float32)
            item["rohsa_components"] = []
            continue

        cutout_header = fits.getheader(cutout_path)
        cutout_wcs = WCS(cutout_header).celestial
        sky = full_wcs.pixel_to_world(float(item["x"]), float(item["y"]))
        x_cut, y_cut = cutout_wcs.world_to_pixel(sky)
        x_idx = int(np.rint(x_cut))
        y_idx = int(np.rint(y_cut))

        model_header = fits.getheader(model_cube_path)
        model_cube = fits.getdata(model_cube_path, memmap=True).astype(np.float32)
        while model_cube.ndim > 3:
            model_cube = model_cube[0]
        if not (0 <= y_idx < model_cube.shape[1] and 0 <= x_idx < model_cube.shape[2]):
            item["rohsa_velocity_kms"] = np.array([], dtype=np.float32)
            item["rohsa_model_total"] = np.array([], dtype=np.float32)
            item["rohsa_components"] = []
            continue

        rohsa_velocity = spectral_axis_kms(model_header, model_cube.shape[0])
        item["rohsa_velocity_kms"] = rohsa_velocity
        item["rohsa_model_total"] = np.asarray(model_cube[:, y_idx, x_idx], dtype=np.float32)

        components = []
        component_params = []
        for idx in range(1, 4):
            amp_path = summary_dir / f"component_{idx:02d}_amplitude.fits"
            mu_path = summary_dir / f"component_{idx:02d}_mu_kms.fits"
            sigma_path = summary_dir / f"component_{idx:02d}_sigma_kms.fits"
            amp = float(fits.getdata(amp_path)[y_idx, x_idx])
            mu = float(fits.getdata(mu_path)[y_idx, x_idx])
            sigma = float(fits.getdata(sigma_path)[y_idx, x_idx])
            if np.isfinite(amp) and np.isfinite(mu) and np.isfinite(sigma) and amp > 0.0 and sigma > 0.0:
                values = amp * np.exp(-0.5 * ((rohsa_velocity - mu) / sigma) ** 2)
                components.append(np.asarray(values, dtype=np.float32))
                component_params.append(
                    {
                        "amp": float(amp),
                        "mu": float(mu),
                        "sigma": float(sigma),
                    }
                )
        item["rohsa_components"] = components
        item["rohsa_component_params"] = component_params


def assign_panel_letters(points: list[dict[str, float | int | str]]) -> None:
    for letter, item in zip(PANEL_LETTERS, points):
        item["panel_letter"] = letter


def add_velocity_windows(
    spectra: list[dict[str, object]],
    mom1_map: np.ndarray,
    mom2_map: np.ndarray,
    velocity_axis: np.ndarray,
    sigma_scale: float,
    min_halfwidth: float,
    max_halfwidth: float,
) -> None:
    axis_min = float(np.nanmin(velocity_axis))
    axis_max = float(np.nanmax(velocity_axis))
    for item in spectra:
        x = int(item["x"])
        y = int(item["y"])
        mom1 = float(mom1_map[y, x])
        mom2 = float(mom2_map[y, x])
        if not np.isfinite(mom1):
            finite = np.asarray(item["velocity_kms"], dtype=float)[np.isfinite(np.asarray(item["spectrum"], dtype=float))]
            mom1 = float(np.nanmedian(finite)) if finite.size else 0.0
        if not np.isfinite(mom2) or mom2 <= 0.0:
            mom2 = min_halfwidth / max(sigma_scale, 1.0)
        halfwidth = float(np.clip(sigma_scale * mom2, min_halfwidth, max_halfwidth))
        v0 = max(axis_min, mom1 - halfwidth)
        v1 = min(axis_max, mom1 + halfwidth)
        if v1 <= v0:
            v0, v1 = axis_min, axis_max
        item["mom1_kms"] = mom1
        item["mom2_kms"] = mom2
        item["xlim_kms"] = (float(v0), float(v1))


def expand_windows_for_rohsa(spectra: list[dict[str, object]]) -> None:
    for item in spectra:
        velocity = np.asarray(item.get("velocity_kms", []), dtype=float)
        finite_velocity = np.isfinite(velocity)
        if velocity.size == 0 or not np.any(finite_velocity):
            continue
        axis_min = float(np.nanmin(velocity[finite_velocity]))
        axis_max = float(np.nanmax(velocity[finite_velocity]))

        support_min = None
        support_max = None

        component_params = list(item.get("rohsa_component_params", []))
        if component_params:
            max_amp = max(float(component["amp"]) for component in component_params)
            amp_floor = max(0.05 * max_amp, 0.003)
            ranges = [
                (
                    float(component["mu"]) - 3.0 * float(component["sigma"]),
                    float(component["mu"]) + 3.0 * float(component["sigma"]),
                )
                for component in component_params
                if float(component["amp"]) >= amp_floor
            ]
            if ranges:
                support_min = min(r[0] for r in ranges)
                support_max = max(r[1] for r in ranges)

        if support_min is None or support_max is None or support_max <= support_min:
            continue

        width = support_max - support_min
        pad = max(10.0, 0.08 * width)
        base_v0 = support_min - pad
        base_v1 = support_max + pad
        center = 0.5 * (base_v0 + base_v1)
        halfwidth = 0.5 * (base_v1 - base_v0) * ROHSA_WINDOW_SCALE
        v0 = max(axis_min, center - halfwidth)
        v1 = min(axis_max, center + halfwidth)
        if v1 <= v0:
            continue
        item["xlim_kms"] = (v0, v1)


def adaptive_ylim(velocity: np.ndarray, spectrum: np.ndarray, xlim_kms: tuple[float, float]) -> tuple[float, float]:
    mask = np.isfinite(velocity) & np.isfinite(spectrum) & (velocity >= xlim_kms[0]) & (velocity <= xlim_kms[1])
    if not np.any(mask):
        mask = np.isfinite(velocity) & np.isfinite(spectrum)
    if not np.any(mask):
        return (-0.05, 0.05)
    values = spectrum[mask]
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return (-0.05, 0.05)
    if ymax == ymin:
        pad = 0.05 * max(abs(ymax), 1.0)
        return (ymin - pad, ymax + pad)
    pad = 0.10 * (ymax - ymin)
    lower = min(0.0, ymin - pad)
    upper = ymax + pad
    return (float(lower), float(upper))


def catalog_highlight(
    catalog_path: Path,
    bubble_id: int,
    header: fits.Header,
) -> dict[str, object]:
    """Return map-pixel samples for one catalog ellipse using its FK5 geometry."""
    catalog = read_catalog(catalog_path)
    rows = catalog[catalog["id"] == bubble_id]
    if rows.empty:
        raise KeyError(f"Bubble id {bubble_id} is not present in {catalog_path}")
    row = rows.iloc[0]
    center = SkyCoord(
        f"{row['ra_hms']} {row['dec_dms']}", unit=(u.hourangle, u.deg)
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 721)
    pa = np.deg2rad(float(row["pa"]))
    semi_major = float(row["maj_as"]) / 2.0
    semi_minor = float(row["min_as"]) / 2.0
    east_arcsec = (
        semi_major * np.sin(pa) * np.cos(angle)
        - semi_minor * np.cos(pa) * np.sin(angle)
    )
    north_arcsec = (
        semi_major * np.cos(pa) * np.cos(angle)
        + semi_minor * np.sin(pa) * np.sin(angle)
    )
    ellipse_sky = center.spherical_offsets_by(east_arcsec * u.arcsec, north_arcsec * u.arcsec)
    wcs = WCS(header).celestial
    x_ellipse, y_ellipse = wcs.world_to_pixel(ellipse_sky)
    x_center, y_center = wcs.world_to_pixel(center)
    return {
        "id": int(row["id"]),
        "x_ellipse": np.asarray(x_ellipse, dtype=float),
        "y_ellipse": np.asarray(y_ellipse, dtype=float),
        "x_center": float(np.ravel(x_center)[0]),
        "y_center": float(np.ravel(y_center)[0]),
    }


def plot_overview(
    nhi_map: np.ndarray,
    mom0_header: fits.Header,
    cone_display_map: dict[str, np.ndarray],
    spectra: list[dict[str, object]],
    output_png: Path,
    output_pdf: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    vmin_nhi: float,
    vmax_nhi: float,
    highlight: dict[str, object] | None = None,
) -> None:
    wcs = WCS(mom0_header)
    norm = Normalize(vmin=vmin_nhi, vmax=vmax_nhi)
    cmap = plt.get_cmap("afmhot").copy()
    cmap.set_bad("black")

    fig = plt.figure(figsize=(14.4, 7.0))
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.35, 0.05, 1.0, 1.0],
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.10,
        wspace=0.18,
        hspace=0.28,
    )
    ax_map = fig.add_subplot(gs[:, 0], projection=wcs)
    right_axes = [
        fig.add_subplot(gs[row, col])
        for row in range(3)
        for col in (2, 3)
    ]

    image = ax_map.imshow(
        nhi_map,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    for sector_name, mask in cone_display_map.items():
        overlay = np.where(mask > 0, 1.0, np.nan)
        ax_map.imshow(
            overlay,
            origin="lower",
            cmap=matplotlib.colors.ListedColormap(["#ffffff"]),
            vmin=0.0,
            vmax=1.0,
            alpha=0.34,
            interpolation="nearest",
        )

    if highlight is not None:
        ax_map.plot(
            np.asarray(highlight["x_ellipse"]),
            np.asarray(highlight["y_ellipse"]),
            color="#101010",
            linewidth=3.6,
            zorder=7,
        )
        ax_map.plot(
            np.asarray(highlight["x_ellipse"]),
            np.asarray(highlight["y_ellipse"]),
            color="#25e6e6",
            linewidth=1.7,
            zorder=8,
        )
        ax_map.text(
            float(highlight["x_center"]),
            float(highlight["y_center"]) + 60.0,
            str(highlight["id"]),
            color="#25e6e6",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=9,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="#101010")],
        )

    for item in spectra:
        ax_map.text(
            float(item["x"]),
            float(item["y"]),
            f"{item['panel_letter']}",
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        )

    ax_map.set_xlim(xlim)
    ax_map.set_ylim(ylim)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("RA (J2000)")
    ax_map.set_ylabel("Dec (J2000)")
    ax_map.coords["ra"].set_ticks(spacing=30 * u.arcmin, color="white")
    ax_map.coords["ra"].display_minor_ticks(True)
    ax_map.coords["ra"].set_minor_frequency(4)
    ax_map.coords["dec"].set_ticks(spacing=15 * u.arcmin, color="white")
    ax_map.coords["dec"].display_minor_ticks(True)
    ax_map.coords["dec"].set_minor_frequency(3)
    ax_map.coords["ra"].set_axislabel("RA (J2000)", color="black")
    ax_map.coords["dec"].set_axislabel("Dec (J2000)", color="black")
    ax_map.coords["ra"].set_ticklabel(color="black")
    ax_map.coords["dec"].set_ticklabel(color="black")
    ax_map.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="white",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )

    ax_cbar = ax_map.inset_axes([0.035, 0.025, 0.045, 0.28])
    cbar = fig.colorbar(image, cax=ax_cbar, orientation="vertical")
    cbar.set_label(r"$N_{\rm HI}$ [$10^{20}\,\rm cm^{-2}$]", fontsize=7)
    cbar.ax.tick_params(labelsize=7, color="white", labelcolor="white", direction="in", length=6)
    cbar.outline.set_edgecolor("white")
    cbar.ax.yaxis.label.set_color("white")
    ax_cbar.set_facecolor("black")

    for ax, item in zip(right_axes, spectra):
        velocity = np.asarray(item["velocity_kms"], dtype=float)
        spectrum = np.asarray(item["spectrum"], dtype=float)
        xlim_local = tuple(item["xlim_kms"])
        ylim_local = adaptive_ylim(velocity, spectrum, xlim_local)
        ax.plot(velocity, spectrum, color=SPECTRUM_COLOR, linewidth=1.2, alpha=0.72)
        rohsa_velocity = np.asarray(item.get("rohsa_velocity_kms", []), dtype=float)
        rohsa_total = np.asarray(item.get("rohsa_model_total", []), dtype=float)
        if rohsa_velocity.size and rohsa_total.size:
            ax.fill_between(
                rohsa_velocity,
                0.0,
                rohsa_total,
                color=ROHSA_TOTAL_COLOR,
                alpha=0.22,
                linewidth=0.0,
                edgecolor="none",
                zorder=2,
            )
            for component in item.get("rohsa_components", []):
                ax.plot(
                    rohsa_velocity,
                    np.asarray(component, dtype=float),
                    color=ROHSA_COMPONENT_COLOR,
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.65,
                    zorder=3,
                )
        ax.axhline(0.0, color="0.35", linewidth=0.7, linestyle="--")
        ax.set_xlim(xlim_local)
        ax.set_ylim(ylim_local)
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        ax.text(
            0.03,
            0.93,
            f"({item['panel_letter']})",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none", "pad": 1.0},
        )
        if item["panel_letter"] == "c":
            legend_handles = [
                Line2D([0], [0], color=SPECTRUM_COLOR, linewidth=1.2, alpha=0.72, label="data spectrum"),
                Patch(facecolor=ROHSA_TOTAL_COLOR, edgecolor="none", alpha=0.22, label="ROHSA total"),
                Line2D([0], [0], color=ROHSA_COMPONENT_COLOR, linewidth=0.9, linestyle="--", alpha=0.65, label="ROHSA components"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=7,
                frameon=True,
                framealpha=0.85,
                facecolor="white",
                edgecolor="0.75",
                borderpad=0.35,
                handlelength=2.0,
                labelspacing=0.3,
            )

    for idx, ax in enumerate(right_axes):
        ax.set_xlabel("velocity (km/s)", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("Jy/beam", fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    required = []
    for sector, directory in DEFAULT_ROHSA_DIRS.items():
        required += [directory / "model_cube.fits", Path(SECTOR_CONFIG[sector]["cutout_path"])]
        for index in range(1, 4):
            required += [directory / f"component_{index:02d}_{field}.fits"
                         for field in ("amplitude", "mu_kms", "sigma_kms")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Final ROHSA overview requires the fit products listed in DATA.md:\n" + "\n".join(missing))
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    iers.conf.auto_download = False

    args = parse_args()
    mom0_path = Path(args.mom0).resolve()
    cube_path = Path(args.cube).resolve()
    mom1_path = Path(args.mom1).resolve()
    mom2_path = Path(args.mom2).resolve()
    fallback_cache_dir = Path(args.midline_cache_dir).resolve()
    r_path = Path(args.r_map).resolve()
    alpha_path = Path(args.alpha_map).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.highlight_catalog is None) != (args.highlight_id is None):
        raise ValueError("Use --highlight-catalog and --highlight-id together.")

    output_png = output_dir / f"{args.output_basename}.png"
    output_pdf = output_dir / f"{args.output_basename}.pdf"
    output_json = output_dir / f"{args.output_basename}_metadata.json"
    if (output_png.exists() or output_pdf.exists() or output_json.exists()) and not args.overwrite:
        raise FileExistsError("Output already exists; rerun with --overwrite to replace it.")

    mom0_data, mom0_header = read_float_fits(mom0_path)
    nhi_map = moment_to_nhi(mom0_data, mom0_header)
    finite_background = np.isfinite(nhi_map)
    mom1_map, _ = read_float_fits(mom1_path)
    mom2_map, _ = read_float_fits(mom2_path)
    r_pc, _ = read_float_fits(r_path)
    alpha_deg, _ = read_float_fits(alpha_path)
    mask_map = {name: read_float_fits(path)[0] for name, path in DEFAULT_MASKS.items()}
    cone_display_map = {
        name: (
            np.asarray(mask, dtype=bool)
            & finite_background
            & np.isfinite(r_pc)
            & (r_pc >= 1000.0 * DISPLAY_R_KPC_RANGE_BY_SECTOR[name][0])
            & (r_pc <= 1000.0 * DISPLAY_R_KPC_RANGE_BY_SECTOR[name][1])
        ).astype(np.float32)
        for name, mask in mask_map.items()
    }

    points = choose_sampling_points(
        r_pc=r_pc,
        alpha_deg=alpha_deg,
        mask_map=mask_map,
        radius_tolerance_pc=float(args.radius_tolerance_pc),
        alpha_weight_pc_per_deg=float(args.alpha_weight_pc_per_deg),
    )
    assign_panel_letters(points)
    velocities, spectra, extracted_now = extract_spectrum(
        cube_path,
        points,
        output_dir,
        fallback_cache_dir=fallback_cache_dir,
        refresh=bool(args.refresh_spectra),
    )
    add_velocity_windows(
        spectra=spectra,
        mom1_map=mom1_map,
        mom2_map=mom2_map,
        velocity_axis=velocities,
        sigma_scale=float(args.velocity_sigma_scale),
        min_halfwidth=float(args.velocity_min_halfwidth),
        max_halfwidth=float(args.velocity_max_halfwidth),
    )
    load_rohsa_models(mom0_header, spectra)
    expand_windows_for_rohsa(spectra)

    for item in spectra:
        item["velocity_min_kms"] = float(np.nanmin(velocities))
        item["velocity_max_kms"] = float(np.nanmax(velocities))

    highlight = None
    if args.highlight_catalog is not None:
        highlight = catalog_highlight(
            Path(args.highlight_catalog).resolve(),
            int(args.highlight_id),
            mom0_header,
        )

    plot_overview(
        nhi_map=nhi_map,
        mom0_header=mom0_header,
        cone_display_map=cone_display_map,
        spectra=spectra,
        output_png=output_png,
        output_pdf=output_pdf,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        vmin_nhi=float(args.vmin_nhi),
        vmax_nhi=float(args.vmax_nhi),
        highlight=highlight,
    )

    metadata = {
        "mom0_path": str(mom0_path),
        "cube_path": str(cube_path),
        "mom1_path": str(mom1_path),
        "mom2_path": str(mom2_path),
        "r_map_path": str(r_path),
        "alpha_map_path": str(alpha_path),
        "mask_paths": {name: str(path) for name, path in DEFAULT_MASKS.items()},
        "output_png": str(output_png),
        "output_pdf": str(output_pdf),
        "selection": {
            "target_radii_kpc_by_sector": {key: list(value["target_radii_kpc"]) for key, value in SECTOR_CONFIG.items()},
            "radius_tolerance_pc": float(args.radius_tolerance_pc),
            "alpha_weight_pc_per_deg": float(args.alpha_weight_pc_per_deg),
            "cone_display_r_kpc_range_by_sector": {
                key: list(value) for key, value in DISPLAY_R_KPC_RANGE_BY_SECTOR.items()
            },
            "cone_display_limited_to_finite_mom0": True,
        },
        "spectra_reextracted_from_cube": bool(extracted_now),
        "spectra_refresh_requested": bool(args.refresh_spectra),
        "highlight": None if highlight is None else {"catalog_id": int(highlight["id"])},
        "spectra": [
            {
                "sector": str(item["sector"]),
                "radius_kpc": float(item["radius_kpc"]),
                "x": int(item["x"]),
                "y": int(item["y"]),
                "panel_letter": str(item["panel_letter"]),
                "r_pc": float(item["r_pc"]),
                "alpha_deg": float(item["alpha_deg"]),
                "mom1_kms": float(item["mom1_kms"]),
                "mom2_kms": float(item["mom2_kms"]),
                "xlim_kms": [float(item["xlim_kms"][0]), float(item["xlim_kms"][1])],
                "finite_channel_count": int(item["finite_channel_count"]),
                "rohsa_component_count": int(len(item.get("rohsa_components", []))),
                "spectrum_text_path": str((output_dir / "sampled_spectra" / f"{item['cache_stem']}.txt").resolve()),
            }
            for item in spectra
        ],
    }
    output_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
