"""Shared paths and the equations used for the final paper catalogue."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "output"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
(RUN_ROOT / "derived").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".local" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".local" / "cache"))
CORRECTED_CATALOG = REPO_ROOT / "code/m31_bubble_catalog_final_yes_118.tab"
CORRECTED_TURB_TABLE = REPO_ROOT / "data/merged_sampling_profile_muH_2p34.tsv"
CORRECTED_BRINKS_TABLE = REPO_ROOT / "data/brinks86_formula_consistent.tsv"
SOURCE_TURB_TABLE = REPO_ROOT / "data/merged_sampling_profile.tsv"
SOURCE_BRINKS_TABLE = REPO_ROOT / "data/brinks+86/brinks86_combined.fwf"
ALPHA_FITS = REPO_ROOT / "data/alpha-m31-jcomb_modHeader.fits"
R_FITS = REPO_ROOT / "data/r-m31-jcomb_modHeader.fits"
MRI_PROFILE = REPO_ROOT / "data/profile_resampled_med.tsv"
NRING_MOM0_FITS = REPO_ROOT / "data/jcomb_vcube_scaled2_mom0.fits"
NRING_ROTATED_MOM0_FITS = REPO_ROOT / "data/jcomb_submed.mom0.-600-45_rotated.fits"
FINAL_REPLAY_ROOT = REPO_ROOT  # Compatibility name; all active paths are public.

PC_TO_CM = 3.0e18
SEC_PER_MYR = 1.0e6 * 3.154e7
MU_H_G = 2.34e-24
SOURCE_TURB_MU_H_G = 1.67e-24
BRINKS_NHI_SCALE = 0.44
BRINKS_GROSS_DIAMETER_SEQ = frozenset({108, 128})


def expansion_time_s(radius_pc, velocity_kms):
    radius_pc = np.asarray(radius_pc, dtype=float)
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    radius_pc, velocity_kms = np.broadcast_arrays(radius_pc, velocity_kms)
    out = np.full(radius_pc.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(radius_pc)
        & np.isfinite(velocity_kms)
        & (radius_pc > 0.0)
        & (velocity_kms > 0.0)
    )
    out[valid] = 0.6 * radius_pc[valid] * PC_TO_CM / (velocity_kms[valid] * 1.0e5)
    return out


def bubble_dote_1e28(radius_pc, velocity_kms, n_hi_cm3):
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    n_hi_cm3 = np.asarray(n_hi_cm3, dtype=float)
    t_exp_s = expansion_time_s(radius_pc, velocity_kms)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (
            0.5
            * MU_H_G
            * n_hi_cm3
            * (velocity_kms * 1.0e5) ** 2
            / t_exp_s
            * 1.0e28
        )
    return rate


def weaver_sn_count(radius_pc, velocity_kms, n_hi_cm3):
    radius_pc = np.asarray(radius_pc, dtype=float)
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    n_hi_cm3 = np.asarray(n_hi_cm3, dtype=float)
    return 1.2 * (velocity_kms / 10.0) ** 2 * (radius_pc / 100.0) ** 3 * n_hi_cm3


def bubble_momentum_cgs(radius_pc, velocity_kms, n_hi_cm3):
    radius_cm = np.asarray(radius_pc, dtype=float) * PC_TO_CM
    velocity_cms = np.asarray(velocity_kms, dtype=float) * 1.0e5
    n_hi_cm3 = np.asarray(n_hi_cm3, dtype=float)
    return (4.0 * np.pi / 3.0) * radius_cm**3 * MU_H_G * n_hi_cm3 * velocity_cms


def corrected_brinks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seq = pd.to_numeric(out.get("Seq"), errors="coerce").to_numpy(dtype=float)
    diameter = pd.to_numeric(out["Diam_pc"], errors="coerce").to_numpy(dtype=float)
    major = pd.to_numeric(out.get("Maj_pc"), errors="coerce").to_numpy(dtype=float)
    minor = pd.to_numeric(out.get("Min_pc"), errors="coerce").to_numpy(dtype=float)
    gross = np.isin(seq, list(BRINKS_GROSS_DIAMETER_SEQ))
    geometric = np.sqrt(major * minor)
    replace = gross & np.isfinite(geometric) & (geometric > 0.0)
    diameter_corrected = diameter.copy()
    diameter_corrected[replace] = np.rint(geometric[replace])

    velocity = np.abs(pd.to_numeric(out["DV_kms"], errors="coerce").to_numpy(dtype=float))
    n_hi_source = pd.to_numeric(out["nHI_cm3"], errors="coerce").to_numpy(dtype=float)
    n_hi_updated = BRINKS_NHI_SCALE * n_hi_source
    radius_pc = 0.5 * diameter_corrected
    t_exp_s = expansion_time_s(radius_pc, velocity)
    rate = bubble_dote_1e28(radius_pc, velocity, n_hi_updated)

    out["Diam_pc_source"] = diameter
    out["Diam_pc_corrected"] = diameter_corrected
    out["diameter_gross_fix"] = replace
    out["radius_pc_corrected"] = radius_pc
    out["v_exp_kms"] = velocity
    out["nHI_source_cm3"] = n_hi_source
    out["nHI_updated_cm3"] = n_hi_updated
    out["t_exp_paper_Myr"] = t_exp_s / SEC_PER_MYR
    out["dot_e_paper_1e-28"] = rate
    return out
