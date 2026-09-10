"""Run final-paper figure entry points; generated files are written to output/."""
from pathlib import Path
import argparse
import subprocess
import sys

SCRIPTS = {
    "1": "fast_vla_jcomb_nHI_maps.py", "2": "r_v_diagram.py",
    "3": "dot_e_balance.py", "4": "build_figure_4.py",
    "S1": "hi_power_spectrum.py", "S2": "build_supplementary_figure_2.py",
    "S3": "consistency_subsets.py", "S4": "a_over_b_vs_r_over_sqrtab.py",
    "S5": "nhi_ring_masks.py", "S6": "vexp_vs_shear_limit.py",
    "S7": "p_per_sn_vs_nsn.py", "S8": "bubble_r_v_age_nsn_vs_r.py",
    "S9": "ob_sn_density_nhi.py", "S10": "vlos_projection_broadening.py",
    "S11": "rohsa_cone_spectra_summary.py", "S12": "hi_profile_major_axis.py",
    "S13": "build_supplementary_figure_13.py", "S14": "h2_hi_ratio_vs_r.py",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*",
                        help="Main figure numbers 1–4 or supplementary S1–S14; default: 1–4")
    parser.add_argument("--all", action="store_true", help="Run all figures; requires all external FITS/ROHSA inputs")
    args = parser.parse_args()
    invalid = [name for name in args.figures if name not in SCRIPTS]
    if invalid:
        parser.error("Unknown figure(s): " + ", ".join(invalid))
    selected = list(SCRIPTS) if args.all else (args.figures or ["1", "2", "3", "4"])
    for name in selected:
        cmd = [sys.executable, str(Path(__file__).resolve().parent / SCRIPTS[name])]
        if name == "S11": cmd += ["--overwrite"]
        subprocess.run(cmd, check=True)
