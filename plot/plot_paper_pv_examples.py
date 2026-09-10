from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

BOOT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_ROOT = BOOT_ROOT / ".local" / "cache"
LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT / "xdg"))
os.environ.setdefault("ASTROPY_CACHE_DIR", str(LOCAL_CACHE_ROOT / "astropy"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from pvextractor import extract_pv_slice
from spectral_cube import SpectralCube

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
ANALYSIS_ROOT = CODE_ROOT / "analysis"
for path in (CODE_ROOT, ANALYSIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bubble_catalog import BubbleRecord, get_bubble_by_id
from bubturb import (
    bubble_pv_plot_v4,
    bubble_pv_plot_v9,
    calculate_relative_endpoints,
    create_pv_path,
    ellipse_line_intersection_improved,
    export_subcube_v3,
    max_min_values_in_ellipse,
)


@dataclass(frozen=True)
class ExampleSpec:
    output_stem: str
    display_label: str
    current_catalog_id: int
    published_v_ang: int

    @property
    def output_name(self) -> str:
        return f"pv_examples_b{self.output_stem}.pdf"


EXAMPLES = (
    ExampleSpec(
        output_stem="10",
        display_label="10",
        current_catalog_id=115,
        published_v_ang=68,
    ),
    ExampleSpec(
        output_stem="17",
        display_label="17",
        current_catalog_id=108,
        published_v_ang=188,
    ),
)


DEFAULT_SUBCUBE_DIRS = (REPO_ROOT / "data" / "pv_subcubes",)

DEFAULT_MAX_SUBCUBE_CENTER_SEPARATION_ARCMIN = 3.0
DEFAULT_LEGACY_TABLE = REPO_ROOT / "initial_script" / "0407-sorted.tab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the two manuscript PV example figures while preserving their "
            "legacy visible labels. By default this follows the app_v7 plotting "
            "path and refuses to cut new subcubes unless explicitly allowed."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "code" / "m31_bubble_catalog_final_yes_118.tab",
    )
    parser.add_argument(
        "--cube",
        type=Path,
        default=REPO_ROOT / "data" / "hi_cube_jcomb_scaled2_submed.fits",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "output" / "pv_examples",
    )
    parser.add_argument(
        "--subcube-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional directory to search for app-style pv-original-###.fits subcubes.",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        metavar="OUTPUT:LABEL:CURRENT_ID:PUBLISHED_VANG",
        help=(
            "Bubble example to render. May be repeated. The default renders the "
            "two manuscript examples as 10:10:115:68 and 17:17:108:188."
        ),
    )
    parser.add_argument(
        "--style",
        choices=("app-v7", "published-v4", "legacy-goodreg-v3"),
        default="app-v7",
        help=(
            "app-v7 uses bubble_pv_plot_v9 as called by code/analysis/app_v7.py. "
            "published-v4 uses the older v4 layout that matches the currently "
            "included PDFs more closely. legacy-goodreg-v3 uses the exact "
            "historical goodreg_20241021 chain: s10.tab parameters, "
            "export_subcube_v2 geometry, and bubble_pv_plot_v3 styling."
        ),
    )
    parser.add_argument(
        "--legacy-table",
        type=Path,
        default=DEFAULT_LEGACY_TABLE,
        help="Old s10.tab copy used by --style legacy-goodreg-v3.",
    )
    parser.add_argument(
        "--legacy-cube",
        type=Path,
        default=REPO_ROOT / "data" / "jcomb_vcube_scaled2.fits",
        help=(
            "Historical full-cube path for --style legacy-goodreg-v3 when "
            "--allow-cut-from-full-cube is explicitly approved."
        ),
    )
    parser.add_argument("--base-angle", type=float, default=38.0)
    parser.add_argument("--scale-factor", type=float, default=2.2)
    parser.add_argument("--velocity-factor", type=float, default=2.0)
    parser.add_argument("--legacy-scale-factor", type=float, default=2.5)
    parser.add_argument("--legacy-v-plot-range", type=float, default=80.0)
    parser.add_argument(
        "--savefig-dpi",
        type=int,
        default=400,
        help="Rasterization resolution for paper PDF image layers.",
    )
    parser.add_argument(
        "--pp-cmap",
        default="gray",
        help="Matplotlib colormap for the two position-position map panels.",
    )
    parser.add_argument(
        "--pv-cmap",
        default="RdYlBu_r",
        help="Matplotlib colormap for the six PV panels.",
    )
    parser.add_argument(
        "--pv-norm",
        choices=("asinh", "linear"),
        default="asinh",
        help="Intensity stretch for the six PV panels.",
    )
    parser.add_argument(
        "--pv-clip-stat",
        choices=("minmax", "p16p84", "window_p05p95"),
        default="minmax",
        help="Statistic used to set PV panel vmin/vmax inside the bubble velocity window.",
    )
    parser.add_argument(
        "--pp-ellipse-color",
        default="y",
        help="Color for bubble ellipses in the two position-position panels.",
    )
    parser.add_argument(
        "--pp-cut-color",
        default="0.5,0.5,0.8,0.9",
        help="Color for PV cut dashed lines in the upper-left position-position panel.",
    )
    parser.add_argument(
        "--pp-angle-label-color",
        default="k",
        help="Color for PV cut angle labels in the upper-left position-position panel.",
    )
    parser.add_argument(
        "--pp-angle-label-stroke-color",
        default="black",
        help='Stroke color for PV cut angle labels; use "none" for no stroke.',
    )
    parser.add_argument(
        "--allow-cut-from-full-cube",
        action="store_true",
        help=(
            "Cut missing subcubes from --cube using export_subcube_v3. Use only "
            "after explicit approval, because this reads the full H I cube."
        ),
    )
    parser.add_argument(
        "--copy-found-subcubes",
        action="store_true",
        help="Copy any found external subcubes into outdir/subcubes before plotting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which subcubes would be used without reading cubes or plotting.",
    )
    parser.add_argument(
        "--max-subcube-center-separation-arcmin",
        type=float,
        default=DEFAULT_MAX_SUBCUBE_CENTER_SEPARATION_ARCMIN,
        help=(
            "Reject candidate subcubes whose FITS WCS center is farther than "
            "this from the target bubble center. This prevents legacy-numbered "
            "subcubes from being silently reused for the wrong sky position."
        ),
    )
    parser.add_argument(
        "--use-current-catalog-vang",
        action="store_true",
        help=(
            "For published-v4, use the current 1113.tab v_ang instead of the "
            "published-PDF marker-angle override."
        ),
    )
    return parser.parse_args()


def parse_example_specs(raw_examples: list[str]) -> tuple[ExampleSpec, ...]:
    if not raw_examples:
        return EXAMPLES

    specs: list[ExampleSpec] = []
    for raw in raw_examples:
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError(
                "--example must have four colon-separated fields: "
                "OUTPUT:LABEL:CURRENT_ID:PUBLISHED_VANG"
            )
        output_stem, display_label, current_id, published_vang = parts
        specs.append(
            ExampleSpec(
                output_stem=output_stem,
                display_label=display_label,
                current_catalog_id=int(current_id),
                published_v_ang=int(float(published_vang)),
            )
        )
    return tuple(specs)


def parse_color(value: str) -> object:
    parts = value.split(",")
    if len(parts) in (3, 4):
        try:
            return tuple(float(part) for part in parts)
        except ValueError:
            return value
    return value


def bubble_to_app_ellipse(
    bubble: BubbleRecord,
    spec: ExampleSpec,
    style: str,
    use_current_catalog_vang: bool,
) -> dict[str, object]:
    v_ang = bubble.v_ang_deg
    if style == "published-v4" and not use_current_catalog_vang:
        v_ang = spec.published_v_ang

    return {
        "id": int(spec.display_label),
        "ra": bubble.ra_deg,
        "dec": bubble.dec_deg,
        "major": int(round(bubble.major_arcsec)),
        "minor": int(round(bubble.minor_arcsec)),
        "pa": int(round(bubble.pa_deg)),
        "velocities": [
            int(round(bubble.vmin_kms)),
            int(round(bubble.vmax_kms)),
        ],
        "collide": int(bubble.collide),
        "v1": int(round(bubble.v1_kms)),
        "v2": int(round(bubble.v2_kms)),
        "v_ang": int(round(v_ang)),
        "good": str(bubble.good),
    }


def legacy_s10_to_ellipse(table_path: Path, spec: ExampleSpec) -> dict[str, object]:
    df = pd.read_fwf(table_path, header=0, infer_nrows=int(1e6))
    match = df[df["id"].astype(int) == int(spec.display_label)]
    if match.empty:
        raise KeyError(f"Could not find legacy id {spec.display_label} in {table_path}")

    row = match.iloc[0]
    coord = SkyCoord(f"{row['ra_hms']} {row['dec_dms']}", unit=(u.hourangle, u.deg))
    collide_raw = row["in_b86"]
    if isinstance(collide_raw, str):
        collide = collide_raw.strip().lower() in {"true", "1", "yes"}
    else:
        collide = bool(collide_raw)
    return {
        "id": int(row["id"]),
        "ra": float(coord.ra.deg),
        "dec": float(coord.dec.deg),
        "major": float(row["major"]) * float(row["sf"]),
        "minor": float(row["minor"]) * float(row["sf"]),
        "pa": float(row["pa"]),
        "velocities": [float(row["v_min"]), float(row["v_max"])],
        "collide": collide,
        "v1": float(row["v1"]),
        "v2": float(row["v2"]),
        "v_ang": float(row["v_ang"]),
        "type": str(row["type"]),
        "source_table": str(table_path),
    }


def candidate_subcube_names(spec: ExampleSpec) -> list[str]:
    legacy_id = int(spec.display_label)
    current_id = spec.current_catalog_id
    return [
        f"pv-original-{legacy_id:03d}.fits",
        f"pv-original-{current_id:03d}.fits",
    ]


def subcube_center_separation_arcmin(
    subcube_path: Path,
    ellipse: dict[str, object],
) -> float:
    header = fits.getheader(subcube_path)
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        raise ValueError(f"{subcube_path} is missing NAXIS1/NAXIS2")

    wcs = WCS(header).celestial
    center_x = naxis1 / 2.0
    center_y = naxis2 / 2.0
    ra_deg, dec_deg = wcs.all_pix2world([[center_x, center_y]], 1)[0]
    target = SkyCoord(float(ellipse["ra"]) * u.deg, float(ellipse["dec"]) * u.deg)
    actual = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg)
    return float(actual.separation(target).to_value(u.arcmin))


def find_subcube(
    spec: ExampleSpec,
    search_dirs: list[Path],
    ellipse: dict[str, object],
    max_center_separation_arcmin: float,
) -> tuple[Path | None, list[dict[str, object]]]:
    rejected: list[dict[str, object]] = []
    for directory in search_dirs:
        for name in candidate_subcube_names(spec):
            candidate = directory / name
            if candidate.exists():
                try:
                    separation = subcube_center_separation_arcmin(candidate, ellipse)
                except Exception as exc:
                    rejected.append(
                        {
                            "candidate": str(candidate),
                            "reason": f"WCS center check failed: {exc}",
                        }
                    )
                    continue
                if separation <= max_center_separation_arcmin:
                    return candidate, rejected
                rejected.append(
                    {
                        "candidate": str(candidate),
                        "reason": "WCS center is too far from target bubble",
                        "center_separation_arcmin": separation,
                        "max_allowed_arcmin": max_center_separation_arcmin,
                    }
                )
    return None, rejected


def export_subcube_legacy_v2(
    cube: SpectralCube,
    ellipse: dict[str, object],
    output_file: Path,
    scale_factor: float,
    v_plot_range: float,
) -> None:
    wcs = cube.wcs.celestial
    x, y = wcs.all_world2pix([[float(ellipse["ra"]), float(ellipse["dec"])]], 1)[0]

    local_scale_factor = scale_factor
    if float(ellipse["major"]) < 80:
        local_scale_factor = 80 / float(ellipse["major"]) * local_scale_factor
    major_axis_pixel = float(ellipse["major"]) / abs(wcs.pixel_scale_matrix[0, 0]) / 3600
    n = int(local_scale_factor * major_axis_pixel)

    x_start = max(int(x - n), 0)
    x_end = min(int(x + n), cube.shape[2])
    y_start = max(int(y - n), 0)
    y_end = min(int(y + n), cube.shape[1])

    velocity_center = float(np.nanmean(ellipse["velocities"]))
    slab = cube.spectral_slab(
        (velocity_center - v_plot_range) * u.km / u.s,
        (velocity_center + v_plot_range) * u.km / u.s,
    )
    slab[:, y_start:y_end, x_start:x_end].write(
        output_file,
        format="fits",
        overwrite=True,
    )


def load_cube(
    cube_path: Path,
    cube_cache: dict[str, SpectralCube],
) -> SpectralCube:
    cache_key = str(cube_path)
    if cache_key not in cube_cache:
        cube_cache[cache_key] = SpectralCube.read(cube_path, format="fits").with_spectral_unit(
            u.km / u.s
        )
    return cube_cache[cache_key]


def bubble_pv_plot_legacy_v3(
    cube: SpectralCube,
    theta: float,
    plot_name: Path,
    ellipse: dict[str, object],
    reg_id: str,
    collide: bool,
) -> None:
    wcs = cube.wcs.celestial
    center_x, center_y = cube.shape[2] / 2, cube.shape[1] / 2
    length = int(np.min([cube.shape[2], cube.shape[1]]) * 0.8)

    endpoints2 = calculate_relative_endpoints(center_x, center_y, theta + 90, length)

    central_slice = cube[int(cube.shape[0] / 2) - 1 : int(cube.shape[0] / 2) + 1, :, :]
    velocity_range_slice = cube.spectral_slab(
        np.nanmin(ellipse["velocities"]) * u.km / u.s,
        np.nanmax(ellipse["velocities"]) * u.km / u.s,
    )

    max_val, _ = max_min_values_in_ellipse(central_slice, ellipse, wcs)
    ell_major = float(ellipse["major"]) / 5.0 * 2.0
    ell_minor = float(ellipse["minor"]) / 5.0 * 2.0
    ell_pa = float(ellipse["pa"]) - 90.0

    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

    ax = plt.subplot(2, 4, 1, projection=wcs)
    cube_moment = central_slice.moment(order=0).value
    ax.imshow(
        cube_moment,
        cmap=plt.cm.gray,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
        norm=mcolors.PowerNorm(gamma=1.0, vmax=max_val, vmin=0),
    )
    ax.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
    ax.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
    ax.coords[0].tick_params(direction="in")
    ax.coords[1].tick_params(direction="in")
    ax.add_patch(
        Ellipse(
            xy=(center_x, center_y),
            width=ell_minor,
            height=ell_major,
            angle=ell_pa,
            edgecolor="y",
            fc="None",
            lw=2,
            ls="--",
        )
    )

    shadow_dist = 0.005
    plt.text(
        0.1 + shadow_dist,
        0.9 - shadow_dist,
        reg_id,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=18,
        color="w" if collide else "k",
        fontweight="bold",
    )
    shadow_dist = 0
    plt.text(
        0.1 + shadow_dist,
        0.9 - shadow_dist,
        reg_id,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=18,
        color="r" if collide else "w",
        fontweight="bold",
    )

    path_list = []
    end_point_list = []
    ang_list = []
    for angle in range(6):
        angle_deg = angle * 30
        ang_list.append(theta + angle_deg)
        endpoints = calculate_relative_endpoints(center_x, center_y, theta + angle_deg, length)
        end_point_list.append(endpoints)
        path_list.append(create_pv_path(endpoints))

    def plot_path(axis, endpoints, color):
        x_coords, y_coords = zip(*endpoints)
        axis.plot(x_coords, y_coords, color=color, ls="--")

    ax = plt.subplot(2, 4, 5, projection=wcs)
    cube_vrange_moment = velocity_range_slice.moment(order=0).value
    max_val_vrange, min_val_vrange = max_min_values_in_ellipse(
        velocity_range_slice,
        ellipse,
        wcs,
    )
    ax.imshow(
        cube_vrange_moment,
        cmap=plt.cm.gray,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
        vmin=min_val_vrange,
        vmax=max_val_vrange,
    )
    ax.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
    ax.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
    ax.coords[0].tick_params(direction="in")
    ax.coords[1].tick_params(direction="in")

    for i, end_point in enumerate(end_point_list):
        plot_path(ax, end_point, color=(0.5, 0.5, 0.8, 0.9))
        plot_pos = np.array(end_point[0]) * 0.9 + np.array(end_point[1]) * 0.1
        ax.text(
            *(plot_pos + [0.2, -0.2]),
            str(ang_list[i]) + "$^\\circ$",
            size=14,
            color=(0.0, 0.0, 0.0, 1),
        )
        ax.text(
            *plot_pos,
            str(ang_list[i]) + "$^\\circ$",
            size=14,
            color=(0.6, 0.6, 0.8, 1),
        )
    plot_path(ax, endpoints2, color=(0.9, 0.1, 0.1, 1.0))
    ax.add_patch(
        Ellipse(
            xy=(center_x, center_y),
            width=ell_minor,
            height=ell_major,
            angle=ell_pa,
            edgecolor="y",
            fc="None",
            lw=2,
            ls="--",
        )
    )

    for i, path in enumerate(path_list):
        pv_diagram = extract_pv_slice(cube, path)
        aspect_ratio = float(pv_diagram.shape[1] / pv_diagram.shape[0])
        ax = plt.subplot(2, 4, 2 + i + int(i > 2), projection=WCS(pv_diagram.header))

        ax.imshow(
            pv_diagram.data,
            aspect=aspect_ratio,
            origin="lower",
            norm=mcolors.PowerNorm(gamma=0.5),
            cmap=plt.cm.RdYlBu_r,
        )
        ax.coords[0].set_format_unit(u.arcmin)
        ax.coords[0].tick_params(direction="in")
        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].set_format_unit(u.km / u.s)
        ax.coords[1].tick_params(direction="in")
        ax.coords[1].display_minor_ticks(True)
        ax.invert_yaxis()

        plt.text(
            0.1,
            0.9,
            str(ang_list[i]) + "$^\\circ$",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            size=14,
        )
        ax.set_ylabel("Velocity [km/s]")
        ax.set_xlabel("Offset [arcmin]")

        if abs(ang_list[i] - float(ellipse["v_ang"])) < 1.0:
            p_center = np.nanmean(ax.get_xlim())
            v_low = (
                float(ellipse["v1"]) * 1e3 - cube.header["CRVAL3"]
            ) / cube.header["CDELT3"] + cube.header["CRPIX3"]
            v_high = (
                float(ellipse["v2"]) * 1e3 - cube.header["CRVAL3"]
            ) / cube.header["CDELT3"] + cube.header["CRPIX3"]
            ax.plot(
                [p_center, p_center],
                [v_low, v_high],
                color="gray",
                ls="--",
                marker="_",
                lw=2,
                markersize=15,
            )

        cdelt3 = cube.header["CDELT3"] * 1e-3
        deltav = float(ellipse["velocities"][0]) - float(ellipse["velocities"][1])
        deltav_pix = abs(deltav / cdelt3)
        v_low = np.nanmean(ax.get_ylim()) - deltav_pix / 2.0
        v_high = np.nanmean(ax.get_ylim()) + deltav_pix / 2.0

        intersect1, intersect2 = ellipse_line_intersection_improved(
            center_x,
            center_y,
            ell_major / 2.0,
            ell_minor / 2.0,
            ell_pa,
            ang_list[i] - 90.0,
        )
        interdist = np.sqrt(np.sum((intersect1 - intersect2) ** 2))
        p_low = np.nanmean(ax.get_xlim()) - interdist / 2.0
        p_high = np.nanmean(ax.get_xlim()) + interdist / 2.0

        ax.plot([p_low, p_low], [v_low, v_high], color="gray", ls="--", marker="_", lw=2, markersize=15)
        ax.plot([p_high, p_high], [v_low, v_high], color="gray", ls="--", marker="_", lw=2, markersize=15)

    plt.savefig(str(plot_name))
    plt.close(fig)


def prepare_subcube(
    spec: ExampleSpec,
    ellipse: dict[str, object],
    args: argparse.Namespace,
    search_dirs: list[Path],
    manifest: dict[str, object],
    cube_cache: dict[str, SpectralCube],
) -> Path:
    found, rejected = find_subcube(
        spec,
        search_dirs,
        ellipse,
        args.max_subcube_center_separation_arcmin,
    )
    if rejected:
        manifest["rejected_subcubes"] = rejected
    cache_dir = args.outdir / "subcubes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.style == "legacy-goodreg-v3":
        cached = cache_dir / f"pv-original-{int(spec.display_label):03d}.fits"
    else:
        cached = cache_dir / f"pv-original-{spec.current_catalog_id:03d}.fits"

    if found is not None:
        if args.copy_found_subcubes and found.resolve() != cached.resolve():
            shutil.copy2(found, cached)
            manifest["subcube_action"] = "copied"
            manifest["source_subcube"] = str(found)
            manifest["subcube"] = str(cached)
            return cached
        manifest["subcube_action"] = "found"
        manifest["source_subcube"] = str(found)
        manifest["subcube"] = str(found)
        return found

    manifest["subcube_action"] = "missing"
    manifest["candidate_names"] = candidate_subcube_names(spec)

    if not args.allow_cut_from_full_cube:
        raise FileNotFoundError(
            "No app-style subcube was found. Re-run only after approval with "
            "--allow-cut-from-full-cube to cut from the full cube."
        )

    if args.style == "legacy-goodreg-v3":
        cube = load_cube(args.legacy_cube, cube_cache)
        export_subcube_legacy_v2(
            cube,
            ellipse,
            output_file=cached,
            scale_factor=args.legacy_scale_factor,
            v_plot_range=args.legacy_v_plot_range,
        )
        manifest["source_cube"] = str(args.legacy_cube)
    else:
        cube = load_cube(args.cube, cube_cache)
        export_subcube_v3(
            cube,
            ellipse,
            scale_factor=args.scale_factor,
            v_plot_factor=args.velocity_factor,
            output_file=str(cached),
        )
        manifest["source_cube"] = str(args.cube)
    manifest["subcube_action"] = "cut_from_full_cube"
    manifest["subcube"] = str(cached)
    return cached


def plot_example(
    spec: ExampleSpec,
    bubble: BubbleRecord,
    ellipse: dict[str, object],
    subcube_path: Path,
    args: argparse.Namespace,
) -> None:
    subcube = SpectralCube.read(subcube_path, format="fits").with_spectral_unit(
        u.km / u.s
    )
    output_path = args.outdir / spec.output_name
    plt.close("all")
    if args.style == "legacy-goodreg-v3":
        bubble_pv_plot_legacy_v3(
            subcube,
            args.base_angle,
            plot_name=output_path,
            ellipse=ellipse,
            reg_id=spec.display_label,
            collide=bool(ellipse["collide"]),
        )
    elif args.style == "app-v7":
        bubble_pv_plot_v9(
            subcube,
            args.base_angle,
            plot_paths=True,
            plot_name=str(output_path),
            ellipse=ellipse,
            reg_id=spec.display_label,
            collide=bubble.collide,
            reg_file=None,
            plot_extra_ellipses=False,
            pp_cmap=args.pp_cmap,
            pv_cmap=args.pv_cmap,
            pv_norm=args.pv_norm,
            pv_clip_stat=args.pv_clip_stat,
            pp_ellipse_color=args.pp_ellipse_color,
            pp_cut_color=parse_color(args.pp_cut_color),
            pp_angle_label_color=args.pp_angle_label_color,
            pp_angle_label_stroke_color=args.pp_angle_label_stroke_color,
            savefig_dpi=args.savefig_dpi,
        )
    else:
        bubble_pv_plot_v4(
            subcube,
            args.base_angle,
            plot_paths=True,
            plot_name=str(output_path),
            ellipse=ellipse,
            reg_id=spec.display_label,
            collide=bubble.collide,
        )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    search_dirs = [*args.subcube_dir, *DEFAULT_SUBCUBE_DIRS]
    examples = parse_example_specs(args.example)
    cube_cache: dict[str, SpectralCube] = {}
    manifests = []

    for spec in examples:
        bubble = get_bubble_by_id(args.catalog, spec.current_catalog_id)
        if args.style == "legacy-goodreg-v3":
            ellipse = legacy_s10_to_ellipse(args.legacy_table, spec)
        else:
            ellipse = bubble_to_app_ellipse(
                bubble,
                spec,
                style=args.style,
                use_current_catalog_vang=args.use_current_catalog_vang,
            )
        manifest: dict[str, object] = {
            "output": str(args.outdir / spec.output_name),
            "display_label": spec.display_label,
            "current_catalog_id": spec.current_catalog_id,
            "catalog": str(args.catalog),
            "center": f"{bubble.ra_hms} {bubble.dec_dms}",
            "style": args.style,
            "pp_cmap": args.pp_cmap,
            "pv_cmap": args.pv_cmap,
            "pv_norm": args.pv_norm,
            "pv_clip_stat": args.pv_clip_stat,
            "ellipse": ellipse,
            "legacy_table": str(args.legacy_table) if args.style == "legacy-goodreg-v3" else None,
            "searched_subcube_dirs": [str(path) for path in search_dirs],
        }

        found, rejected = find_subcube(
            spec,
            search_dirs,
            ellipse,
            args.max_subcube_center_separation_arcmin,
        )
        if args.dry_run:
            manifest["subcube_action"] = "found" if found else "missing"
            manifest["subcube"] = str(found) if found else None
            manifest["candidate_names"] = candidate_subcube_names(spec)
            if rejected:
                manifest["rejected_subcubes"] = rejected
            manifests.append(manifest)
            continue

        try:
            subcube_path = prepare_subcube(
                spec,
                ellipse,
                args,
                search_dirs,
                manifest,
                cube_cache,
            )
        except FileNotFoundError as exc:
            manifest["error"] = str(exc)
            manifests.append(manifest)
            continue
        plot_example(spec, bubble, ellipse, subcube_path, args)
        manifest["plotted"] = True
        manifests.append(manifest)

    metadata_path = args.outdir / "pv_examples_metadata.json"
    metadata_path.write_text(json.dumps(manifests, indent=2) + "\n")

    failures = [item for item in manifests if item.get("error")]
    if failures:
        print(json.dumps(manifests, indent=2))
        raise SystemExit(2)

    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
