#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OB mass formation rate per custom age bins (Msun/yr)
----------------------------------------------------
- Input: Kang+ (2009) table2.dat
- Compute total M_02 (stellar mass) per user-defined age bin
  and divide by bin width (converted to years).

AGE_BINS can be non-uniform.
Example: AGE_BINS = (0., 4., 10., 100., 400.)

Output:
  - Printed summary
  - CSV file: ob_age_mass_rate.csv
"""

import numpy as np
import pandas as pd

# ---------- Kang+2009 fixed-width reader ----------
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
    # sentinel values to NaN
    for c in ["Age_02","b_Age_02","B_Age_02","Age_05","b_Age_05","B_Age_05"]:
        df.loc[np.isclose(df[c], -99.0, equal_nan=False), c] = np.nan
    for c in ["M_02","b_M_02","B_M_02","M_05","b_M_05","B_M_05"]:
        df.loc[np.isclose(df[c], -9.9e+01, equal_nan=False), c] = np.nan
    return df

# ---------- core computation ----------
def mass_rate_by_age_bins(
    ages_myr: np.ndarray,
    masses_msun: np.ndarray,
    age_bins: list
) -> pd.DataFrame:
    """
    Compute total M_02 per age bin, divide by bin width (in yr).
    Output rate in Msun/yr.
    """
    ages = np.asarray(ages_myr, float)
    masses = np.asarray(masses_msun, float)
    mask = np.isfinite(ages) & np.isfinite(masses)
    ages, masses = ages[mask], masses[mask]

    cats = pd.cut(ages, bins=age_bins, right=False, include_lowest=True)
    df = pd.DataFrame({"age": ages, "M02": masses, "bin": cats})
    gb = df.groupby("bin", observed=True)
    mass_sum = gb["M02"].sum(min_count=1)
    count = gb["M02"].count()

    lefts  = [iv.left  for iv in mass_sum.index]
    rights = [iv.right for iv in mass_sum.index]
    width_myr = np.asarray(rights) - np.asarray(lefts)
    width_yr = width_myr * 1e6  # convert to years

    rate_msun_per_yr = mass_sum.to_numpy() / width_yr

    out = pd.DataFrame({
        "age_left_Myr": lefts,
        "age_right_Myr": rights,
        "bin_width_Myr": width_myr,
        "count": count.to_numpy(),
        "mass_sum_Msun": mass_sum.to_numpy(),
        "mass_rate_Msun_per_yr": rate_msun_per_yr
    })
    return out

# ---------- print ----------
def print_table(df: pd.DataFrame):
    print("\n=== OB mass formation rate per age bin (M_02 sum / Δt) ===")
    for _, r in df.iterrows():
        l, rgt = r["age_left_Myr"], r["age_right_Myr"]
        w = r["bin_width_Myr"]
        cnt = int(r["count"])
        msum = r["mass_sum_Msun"]
        mrate = r["mass_rate_Msun_per_yr"]
        print(f"[{l:6.1f}, {rgt:6.1f}) Myr | width={w:6.1f} | N={cnt:4d} | "
              f"ΣM_02={msum:10.3g} Msun | rate={mrate:10.3g} Msun/yr")
    print("===========================================================\n")

# ---------- main ----------
def main(
    kang_path="../code/kang_09_table2.dat",
    AGE_BINS=(0., 4., 10., 100., 400.)
):
    df = read_kang2009_fwf(kang_path)
    res = mass_rate_by_age_bins(df["Age_02"].to_numpy(), df["M_02"].to_numpy(), list(AGE_BINS))
    print_table(res)
    res.to_csv("ob_age_mass_rate.csv", index=False)
    print("Saved: ob_age_mass_rate.csv")

if __name__ == "__main__":
    main()