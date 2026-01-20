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
- `num2tex`: unknown
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
