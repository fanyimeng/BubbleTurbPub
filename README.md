# M31 H I Superbubbles and Turbulence

This repository contains analysis scripts, data tables, and LaTeX sources used to study superbubbles and turbulence in the M31 H I disk with combined FAST and JVLA data. 

## Contents
- `plot/` Python figure scripts and generated PDFs.
- `code/` Bubble and ring parameter tables and helper scripts.
- `data/` Input tables and expected FITS cubes and maps that are not versioned.
- `paper/` LaTeX source and bibliography style files.
- `other/` Supplementary materials.

## FITS files
- Datacube (18 GB) that used for generate the FITS files for plotting and caculation: https://gofile.me/7L8Ih/zWZetZ2dA  **The password for downloading is the manuscript number (looks like `202x-xx-xxxxx`). Please use the real manuscript number.**
- Other FITS files that are soft-linked-to in `data/` is zipped can placed in: https://gofile.me/7L8Ih/0zxCBEDZl **The password for downloading is the manuscript number (looks like `202x-xx-xxxxx`). Please use the real manuscript number.**

## Data and file expectations
- Large FITS cubes and auxiliary maps are not tracked in git.
- Place (in most cases replace the softlinks with same name) required FITS inputs (can be downloaded from ...) under `data/` using the filenames referenced in each script.
- Data tables and other smaller files are already in this repo.

## Software
- Python 3 with `numpy`, `pandas`, `astropy`, `matplotlib`.
- The image combine tool `J-comb` (as described in the manuscript's Methods part) is adopted from https://github.com/SihanJiao/J-comb with parameters specified in Methods. For details, see https://ui.adsabs.harvard.edu/abs/2022SCPMA..6599511J/abstract

## Reproducing figures
- Run scripts from the repository root, for example:
  - `python plot/dot_e_balance.py`
- Most plotting script writes `plot/<script_stem>.pdf` by default.


