# Script Overview

This document summarizes what each script in this repository does and which
parameter tables drive the plotting. Run scripts from the repo root unless a
script hard-codes absolute paths.

## Plotting Parameter Tables (Key Inputs)

- `code/1113.tab`: primary fixed-width bubble catalog used by most plot scripts.
  It contains geometry and derived columns (for example `radius_pc`,
  `expansion_vel`, `r_kpc`, `SN_weaver`, `n_HI_ring_cm-3`). In the current repo
  this file effectively serves as both the bubble table and the ring parameter
  table for plots.
- `data/profile_resampled_max.tsv`, `data/profile_resampled_min.tsv`:
  resampled radial profiles (mom0/mom2) used for turbulence curves and
  comparisons in multiple dotE/v2 plots.
- `data/profile_resampled_med.tsv`: median profile used for the MRI curve in
  the energy budget plots.
- `data/profile_output_VLA_JCOMB.txt`: major-axis profile used by
  `plot/hi_profile_major_axis.py`.
- `data/brinks+86/table2.dat` and `data/brinks+86/table3.dat`:
  Brinks+86 source tables; combined outputs in
  `data/brinks+86/brinks86_combined.{csv,fwf}` are used in comparison plots.
- `code/kang_09_table2.dat`: Kang+2009 OB association table used in
  `plot/ob_sn_density_nhi.py`.

Notes:
- Some scripts still reference `code/1110.tab` or `code/0709-decon_hb.tab`,
  which are not present in this repo. For current runs, point those parameters
  to `code/1113.tab` or supply the legacy files.
- `plot/nhi_ring_masks.py` can compute and append ring densities to a fixed
  width catalog and is the place to update `n_HI_ring_cm-3`.

## Scripts by Folder

### code/

- `code/brinks86_combine.py`: merges Brinks+86 table2/table3 into
  `data/brinks+86/brinks86_combined.{csv,fwf}` with headers.
- `code/brinks86_combine_with_regions.py`: same merge as above, but also attaches
  DS9 pixel coordinates from `data/brinks+86/b86_inpix_id.reg` and writes
  `data/brinks+86/brinks86_combined_reg.{csv,fwf}`.

### plot/

- `plot/a_over_b_vs_r_over_sqrtab.py`: two-panel figure; analytic projection
  error (left) and KDE of axis ratios from `code/1113.tab` (right). Outputs
  `a_over_b_vs_r_over_sqrtab.pdf`.
- `plot/bubble_r_v_age_nsn_vs_r.py`: five-panel bubble properties vs radius
  (r, vexp, age, n_HI, N_SN) using `code/1113.tab` and radius/alpha/mom0 FITS.
  Outputs `bubble_r_v_age_nsn_vs_r.pdf`.
- `plot/dot_e_balance.py`: energy budget vs R with turbulence band, bubbles, and
  optional MRI curve; uses `code/1113.tab`, ring density, and resampled
  profiles. Outputs `dot_e_balance.pdf`.
- `plot/dot_e_vs_timescale.py`: bubble dotE vs expansion timescale with Brinks
  comparison; uses `code/1113.tab`, `data/brinks+86/brinks86_combined.fwf`, and
  radius/alpha FITS. Outputs `dot_e_vs_timescale.pdf`.
- `plot/dot_e_vs_timescale_insets.py`: same as above with an inset for the full
  Brinks sample. Outputs `dot_e_vs_timescale_insets.pdf`.
- `plot/dot_e_vs_turb.py`: scatter of bubble dotE vs local turbulence dotE using
  `code/1113.tab` and resampled profiles. Outputs `dot_e_vs_turb.pdf`.
- `plot/fast_vla_jcomb_nHI_maps.py`: three-panel N_HI maps (FAST, VLA, JCOMB)
  from FITS inputs. Outputs `fast_vla_jcomb_nHI_maps.pdf`.
- `plot/h2_hi_ratio_vs_r.py`: radial H2/HI ratio band with bubble ring medians
  from `code/1113.tab`. Outputs `h2_hi_ratio_vs_r.pdf`.
- `plot/hi_power_spectrum.py`: FFT-based power spectrum comparison for FAST,
  VLA, and combined moment0 FITS. Outputs `hi_power_spectrum.pdf`.
- `plot/hi_profile_major_axis.py`: multi-panel major-axis profile plus image
  cutout using `data/profile_output_VLA_JCOMB.txt` and a rotated mom0 FITS.
  Outputs `hi_profile_major_axis.pdf`.
- `plot/main_chart_sum.py`: full energy budget figure (configurable) with
  turbulence band and bubble coloring. Defaults reference legacy tables; use
  `code/1113.tab` for current runs. Outputs `EnergyBudget_pub_ALL.pdf`.
- `plot/nhi_ring_masks.py`: computes `n_HI_center_cm-3`, `n_HI_ellipse_cm-3`,
  and `n_HI_ring_cm-3` per bubble and can write an updated fixed-width catalog;
  also creates a preview PDF. Outputs `nhi_ring_masks.pdf` and optional table.
- `plot/ob_sn_density_nhi.py`: OB association volume density vs radius with
  bubble SN density scatter; uses `code/kang_09_table2.dat` and `code/1113.tab`.
  Outputs `ob_sn_density_nhi.pdf`.
- `plot/p_per_sn_vs_nsn.py`: scatter of p_per_SN vs N_SN using Weaver/Chevalier
  selection from `code/1113.tab`. Outputs `p_per_sn_vs_nsn.pdf`.
- `plot/r_v_diagram.py`: five-panel R-V diagram with rotated mom0 background,
  bubble overlays, and Brinks regions. Outputs `r_v_diagram.pdf`.
- `plot/v2_over_t_balance.py`: energy budget analog using v^2/t for bubbles and
  turbulence; uses `code/1113.tab` and resampled profiles. Outputs
  `v2_over_t_balance.pdf`.
- `plot/v2_over_t_vs_turb.py`: scatter of bubble v^2/t vs turbulent v^2/t using
  `code/1113.tab` and resampled profiles. Outputs `v2_over_t_vs_turb.pdf`.
- `plot/v2_vs_turb.py`: scatter of bubble v^2 vs turbulent v^2 at matched radius
  using `code/1113.tab` and resampled profiles. Outputs `v2_vs_turb.pdf`.
- `plot/vexp_vs_shear_limit.py`: compares expansion velocity to projected shear
  limit; uses `code/1113.tab` plus alpha/r/mom0 FITS. Outputs
  `vexp_vs_shear_limit.pdf` and optional CSV/TXT.
- `plot/vlos_projection_broadening.py`: analytic vs observed line-of-sight
  broadening with shared colorbar and N_HI contours. Outputs
  `vlos_projection_broadening.pdf`.

### data/

- `data/CAR_B05_MP_C0402_1222test3_simobs_casa.py`: CASA simobserve + flagging
  + tclean script for the CAR_B05_MP_C0402_1222test3 simulation. Produces
  measurement sets and FITS images under the configured external workdir.

### repo utilities

- `git_push.sh`: convenience script to `git add`, `git commit`, and `git push`
  to `origin main` with a provided commit message.
