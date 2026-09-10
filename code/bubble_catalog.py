from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


@dataclass(frozen=True)
class BubbleRecord:
    id: int
    ra_deg: float
    dec_deg: float
    ra_hms: str
    dec_dms: str
    major_arcsec: float
    minor_arcsec: float
    pa_deg: float
    vmin_kms: float
    vmax_kms: float
    v1_kms: float
    v2_kms: float
    v_ang_deg: float
    collide: int
    good: str

    @property
    def velocity_bounds_sorted(self) -> tuple[float, float]:
        return tuple(sorted((self.vmin_kms, self.vmax_kms)))

    @property
    def preferred_velocity_bounds_sorted(self) -> tuple[float, float]:
        return tuple(sorted((self.v1_kms, self.v2_kms)))


def read_catalog(catalog_path: str | Path) -> pd.DataFrame:
    return pd.read_fwf(catalog_path, header=0, infer_nrows=int(1e6))


def row_to_bubble(row: pd.Series) -> BubbleRecord:
    coord = SkyCoord(
        f"{row['ra_hms']} {row['dec_dms']}",
        unit=(u.hourangle, u.deg),
    )
    return BubbleRecord(
        id=int(row["id"]),
        ra_deg=float(coord.ra.deg),
        dec_deg=float(coord.dec.deg),
        ra_hms=str(row["ra_hms"]),
        dec_dms=str(row["dec_dms"]),
        major_arcsec=float(row["maj_as"]),
        minor_arcsec=float(row["min_as"]),
        pa_deg=float(row["pa"]),
        vmin_kms=float(row["vmin_kms"]),
        vmax_kms=float(row["vmax_kms"]),
        v1_kms=float(row["v1"]),
        v2_kms=float(row["v2"]),
        v_ang_deg=float(row["v_ang"]),
        collide=int(row["collide"]),
        good=str(row["good"]),
    )


def get_bubble_by_id(catalog_path: str | Path, bubble_id: int) -> BubbleRecord:
    df = read_catalog(catalog_path)
    match = df[df["id"] == bubble_id]
    if match.empty:
        raise KeyError(f"Bubble id {bubble_id} not found in {catalog_path}")
    return row_to_bubble(match.iloc[0])
