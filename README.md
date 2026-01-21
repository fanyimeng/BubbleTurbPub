# M31 H I Superbubbles and Turbulence

This repository contains the analysis scripts and figure-generation code for the
M31 H I superbubble and turbulence study.

## Compiled Standalone Software and/or Source Code

- Source code only (no compiled binaries).
- Primary scripts are in `plot/` and helper code is in `code/`.

## System Requirements

Operating systems:
- macOS 15.3 (arm64) as tested on this machine.

Tested versions (this machine):
- macOS 15.3 (arm64)
- Python 3.12.2 (conda-forge)

Non-standard hardware:
- None required. A machine with >20 GB free disk space is recommended if you
  download the full FITS dataset.

Software dependencies (installed versions on this machine):
- Python 3.12.2
- `numpy`: 1.26.4
- `pandas`: 2.2.2
- `astropy`: 7.0.1
- `matplotlib`: 3.9.2
- `scipy`: 1.13.1
- `seaborn`: 0.13.2
- `tqdm`: 4.66.5
- Optional: CASA (not installed; only needed for
  `data/CAR_B05_MP_C0402_1222test3_simobs_casa.py`)

To reprint versions on your machine:
```bash
python scripts/print_versions.py
```

## Installation Guide

Instructions (from repo root):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas astropy matplotlib scipy seaborn num2tex tqdm
```

Typical install time on a normal desktop computer:
- 5 to 10 minutes (depending on network speed and whether wheels are available).

## Demo

Most scripts are lightweight and read small tabular inputs (for example
`code/1113.tab`), so they can be used as a demo without the large FITS files.

Demo dataset:
- `data/demo_bubbles.tab` (small subset of bubble parameters).

Demo command (no file edits required):
```bash
python -c "import sys; sys.path.append('plot'); import p_per_sn_vs_nsn as m; m.INPUT_TAB='data/demo_bubbles.tab'; m.OUTPUT_PDF='plot/demo_p_per_sn_vs_nsn.pdf'; m.main()"
```

Expected output:
- `plot/demo_p_per_sn_vs_nsn.pdf`
- Console line: `Saved: plot/demo_p_per_sn_vs_nsn.pdf`

Expected runtime on a normal desktop computer:
- < 5 seconds.

## Instructions for Use

1. Run scripts from the repository root (examples in `SCRIPTS.md`).
2. Many plots require FITS inputs that are not versioned in git. Download and
   place them under `data/` with the filenames referenced in each script.
3. Execute the plot script you need, for example:
   ```bash
   python plot/dot_e_balance.py
   ```

Optional reproduction instructions (full paper figures):
- See `SCRIPTS.md` for the mapping between paper figures and scripts.
- FITS data sources used for the paper:
  - Datacube (18 GB): https://gofile.me/7L8Ih/zWZetZ2dA
  - Other FITS files (zipped): https://gofile.me/7L8Ih/0zxCBEDZl
  - Password: the manuscript number (format like `202x-xx-xxxxx`).

## Paper Figures and Scripts (Concise Index)

Figures are listed in the order they appear in `paper/main.tex`.

| Paper label | PDF in `plot/` | Generator script |
| --- | --- | --- |
| `f.three` | `fast_vla_jcomb_nHI_maps.pdf` | `plot/fast_vla_jcomb_nHI_maps.py` |
| `f.momentzero` | `r_v_diagram.pdf` | `plot/r_v_diagram.py` |
| `f.mainchart` (a) | `dot_e_balance.pdf` | `plot/dot_e_balance.py` |
| `f.mainchart` (b) | `v2_over_t_balance.pdf` | `plot/v2_over_t_balance.py` |
| `f.mainchart` (c) | `dot_e_vs_turb.pdf` | `plot/dot_e_vs_turb.py` |
| `f.mainchart` (d) | `dot_e_vs_timescale_insets.pdf` | `plot/dot_e_vs_timescale_insets.py` |
| `f.combineuv` | `hi_power_spectrum.pdf` | `plot/hi_power_spectrum.py` |
| `f.aoverb` | `a_over_b_vs_r_over_sqrtab.pdf` | `plot/a_over_b_vs_r_over_sqrtab.py` |
| `f.nhi_ring_preview` | `nhi_ring_masks.pdf` | `plot/nhi_ring_masks.py` |
| `f.nsf` | `vexp_vs_shear_limit.pdf` | `plot/vexp_vs_shear_limit.py` |
| `f.momentum` | `p_per_sn_vs_nsn.pdf` | `plot/p_per_sn_vs_nsn.py` |
| `f.rvage` | `bubble_r_v_age_nsn_vs_r.pdf` | `plot/bubble_r_v_age_nsn_vs_r.py` |
| `f.obsne` | `ob_sn_density_nhi.pdf` | `plot/ob_sn_density_nhi.py` |
| `f.deltav_model` | `vlos_projection_broadening.pdf` | `plot/vlos_projection_broadening.py` |
| `f.m31_profile` | `hi_profile_major_axis.pdf` | `plot/hi_profile_major_axis.py` |
| `f.moltohiratio` | `h2_hi_ratio_vs_r.pdf` | `plot/h2_hi_ratio_vs_r.py` |

Key plotting tables (short list):
- `code/1113.tab`: main bubble catalog used by most plots; also the ring table.
- `data/profile_resampled_{max,min,med}.tsv`: resampled radial profiles.
- `data/profile_output_VLA_JCOMB.txt`: major-axis profile for `plot/hi_profile_major_axis.py`.
- `data/brinks+86/brinks86_combined.fwf`: Brinks+86 comparison table.
- `code/kang_09_table2.dat`: OB-star census table for `plot/ob_sn_density_nhi.py`.

Other non-figure scripts:
- `code/brinks86_combine.py`, `code/brinks86_combine_with_regions.py`: build Brinks+86 tables.

## Initial Script Folder (Legacy Catalog Build)

The `initial_script/` folder holds early, ad-hoc scripts used to assemble the
HI bubble catalog, extract subcubes, and generate PV plots. Most scripts call
the shared library `initial_script/bubturb.py`.

Common inputs and formats:
- Fixed-width tables (`.tab`) read via `pd.read_fwf(...)` and written back with
  `DataFrame.to_string(...)` or `tabulate`.
- DS9 region files (`.reg`) for ellipse visualization and cross-checks.
- Spectral cube FITS read via `spectral_cube.SpectralCube` and plotted with
  `pvextractor`.

Key drivers and utilities (non-exhaustive):
- `initial_script/app_v7.py`: Streamlit UI for interactive review/editing.
- `initial_script/cc00_dfProcess.py`: table conversion pipeline driver.
- `initial_script/cc01_subplot.py` / `initial_script/cc01_subplot_MP.py`: batch
  PV plotting (single-process and multiprocessing).
- `initial_script/CC03_trivial_parameters.py`: derived kinematic/size columns.
- `initial_script/CC05_size_deconvolve.py`: beam deconvolution for radii.
- `initial_script/ds9_2_pandas.py` and `initial_script/pandas_2_ds9.py`: DS9
  region conversion utilities.

Typical data flow (example):
1. DS9 regions -> fixed-width table (`ds9_2_pandas.py` or `bubturb.reg2df`).
2. Normalize ellipse params, add RA/Dec strings, convert axes to arcsec.
3. Reassign IDs and curate the table (`cc00_dfProcess.py`).
4. Extract subcubes and generate PV plots (`cc01_subplot_MP.py`).
5. Review/edit via `app_v7.py`, then compute derived quantities.
