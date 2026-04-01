"""
Combine Brinks & Bajaja (1986) HI hole tables into a single headered table.

Inputs
------
- table2_path : Path to the observed-properties table (default: data/brinks+86/table2.dat).
- table3_path : Path to the derived-properties table (default: data/brinks+86/table3.dat).

Outputs
-------
- data/brinks+86/brinks86_combined.csv : CSV with explicit headers and merged columns.
- data/brinks+86/brinks86_combined.fwf : Fixed-width text aligned for pandas.read_fwf.

Notes
-----
- All original fields are preserved as strings; numeric columns are best-effort cast when writing CSV.
- Missing entries remain empty so pandas will treat them as NaN on read.
- Seq is kept as int for a reliable merge key.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "brinks+86"


TABLE2_COLUMNS: List[str] = [
    "Seq",
    "Xpos_arcmin",
    "Ypos_arcmin",
    "RADEC_B1950",
    "HRV_kms",
    "DV_kms",
    "FWHM_minor_pc",
    "FWHM_major_pc",
    "psi_deg",
    "INT_K",
    "Contr",
    "Q",
    "T",
    "C",
    "Remarks",
]


TABLE3_COLUMNS: List[str] = [
    "Seq",
    "R_kpc",
    "theta_deg",
    "Maj_pc",
    "Min_pc",
    "PA_deg",
    "Diam_pc",
    "Ratio",
    "nHI_cm3",
    "Age_Myr",
    "Mass_1e4Msun",
    "Energy_1e50erg",
]


COMBINED_COLUMNS: List[str] = TABLE2_COLUMNS + [
    col for col in TABLE3_COLUMNS if col != "Seq"
]


def _read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read a pipe-delimited table into a DataFrame with trimmed string columns."""
    df = pd.read_csv(
        path,
        sep="\\|",
        engine="python",
        names=columns,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    for col in columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        df.loc[df[col] == "", col] = pd.NA
    df["Seq"] = pd.to_numeric(df["Seq"], errors="coerce").astype("Int64")
    return df


def _format_fwf(df: pd.DataFrame, widths: Dict[str, int]) -> str:
    """Return a fixed-width text block for the provided DataFrame."""
    pieces: List[str] = []
    header = " ".join(f"{col:<{widths[col]}}" for col in df.columns)
    pieces.append(header)
    for _, row in df.iterrows():
        fields: List[str] = []
        for col in df.columns:
            val = "" if pd.isna(row[col]) else str(row[col])
            fields.append(f"{val:<{widths[col]}}")
        pieces.append(" ".join(fields))
    return "\n".join(pieces)


def _suggest_widths(df: pd.DataFrame) -> Dict[str, int]:
    """Compute per-column widths for fixed-width output."""
    widths: Dict[str, int] = {}
    for col in df.columns:
        max_len = max(len(str(v)) for v in df[col].fillna(""))
        widths[col] = max(max_len, len(col))
    return widths


def merge_tables(
    table2_path: Path = DATA_DIR / "table2.dat",
    table3_path: Path = DATA_DIR / "table3.dat",
    output_csv: Path = DATA_DIR / "brinks86_combined.csv",
    output_fwf: Path = DATA_DIR / "brinks86_combined.fwf",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge Brinks+86 tables and write CSV and fixed-width outputs."""
    print(f"Reading {table2_path}")
    t2 = _read_table(table2_path, TABLE2_COLUMNS)
    print(f"Reading {table3_path}")
    t3 = _read_table(table3_path, TABLE3_COLUMNS)

    merged = pd.merge(t2, t3, on="Seq", how="outer", suffixes=("", "_t3"))
    merged = merged[COMBINED_COLUMNS]

    print(f"Writing CSV to {output_csv}")
    merged.to_csv(output_csv, index=False)

    widths = _suggest_widths(merged)
    fwf_text = _format_fwf(merged, widths)
    print(f"Writing fixed-width table to {output_fwf}")
    output_fwf.write_text(fwf_text + "\n", encoding="ascii")

    return t2, t3, merged


if __name__ == "__main__":
    merge_tables()
