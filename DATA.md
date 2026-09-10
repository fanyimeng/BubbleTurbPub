# Input data and placement

All paths below are relative to the repository root. Keep the original FITS
headers, pixel grids and units. The plotting scripts do not silently resample
data to repair a mismatched WCS. Large FITS files and local symlinks are ignored
by Git.

## Included small inputs

| File | Role |
| --- | --- |
| `code/m31_bubble_catalog_final_yes_118.tab` | Final physical catalogue; see `code/CATALOG.md` |
| `data/merged_sampling_profile_muH_2p34.tsv` | Final ROHSA turbulence radial profile and energy rates |
| `data/merged_sampling_profile.tsv` | Source profile before the mean-mass normalization correction |
| `data/merged_mri_profile.tsv` | ROHSA density/profile support table |
| `data/profile_resampled_med.tsv` | MRI profile used by the final energy-budget figure |
| `data/profile_resampled_min.tsv`, `data/profile_resampled_max.tsv` | Retained alternative profile inputs |
| `data/brinks+86/brinks86_combined.fwf` | Published comparison input; corrections applied in `paper_formula_common.py` |
| `data/brinks86_formula_consistent.tsv` | Precomputed final comparison quantities |
| `data/consistency_fm_points.tsv`, `data/consistency_rl_points.tsv` | Frozen plotted values for Supplementary Figure 3 |
| `code/kang_09_table2.dat` | OB-star census input |
| `data/profile_output_VLA_JCOMB.txt` | Major-axis profile input |
| `data/Source_Data.xlsx`, `data/Supplementary_Data.xlsx` | Final submitted source-data workbooks |

The consistency tables contain only the identifiers, plotted quantities,
uncertainties and membership flags needed for the final diagnostic. They are
not new classifications or a replacement for the candidate archive.

## Image inputs

Start with the [Tsinghua data archive](https://cloud.tsinghua.edu.cn/d/459ac2c2c8c847e68781/)
and its restoration instructions. The following list is the code's required
layout; it is not a claim that every later derived product is included in the
original archive. The Git-only demos are S3, S4 and S7.

| Required under `data/` | Figures |
| --- | --- |
| `fast_mom0_rotated.fits`, `vla_submed.mom0.-600-45_rotated.fits`, `jcomb_submed.mom0.-600-45_rotated.fits` | 1; the combined map also 2 and S5 |
| `b302_-94-88.fits`, `b302_-88-82.fits`, `b302_-82-76.fits` | 2 |
| `alpha-m31-jcomb_modHeader.fits`, `r-m31-jcomb_modHeader.fits` | 3, 4, S5–S6, S8–S9, S11, S13–S14, as applicable |
| `all-jcomb.fits.moment.integrated.fits`, `contsub-02.fits.moment.integrated.fits`, `fast-rg.fits.moment.integrated.fits` | S1 |
| `jcomb_vcube_scaled2_mom0.fits` | S5, S6, S8, S9 |
| `vlos_delta_abs.fits`, `jcomb_submed.mom0.-600-45.fits` | S10; the latter also S11 |
| `bubbleTurbCube_x1beam_-700-100_0.006_mom0_rotated.fits` | S12 |
| `M31_H2_to_HI_ratio.fits` | S14 |

`data/b86_inpix_id.reg` and the Brinks tables needed by Figure 2 are included.
The original analysis geometry is required: the angle map is in degrees and the
radius map is in pc. Some catalogue columns use historical radius conventions;
the scripts preserve the exact convention of each final figure.

## PV examples (S2)

Place the WCS-validated `pv-original-115.fits` and `pv-original-108.fits` in
`data/pv_subcubes/`. Older files bearing figure labels 010/017 are not
interchangeable with these current-catalogue targets.

If those subcubes are unavailable, supply the original
`data/hi_cube_jcomb_scaled2_submed.fits` and run the PV generator with
`--allow-cut-from-full-cube` in addition to the display arguments in
`plot/build_supplementary_figure_2.py`. The generated cuts are written under
`output/`; the full cube is not modified.

## ROHSA overview (S11)

The existing decomposition products are required for this plot. In addition to
the combined moment-0 and geometry maps, prepare:

```text
data/hi_cube_jcomb_scaled2_submed.fits
data/jcomb_vcube_scaled2_submed.fits.moment.weighted_coord
data/jcomb_vcube_scaled2_submed.fits.moment.weighted_dispersion_coord
data/cone_masks/m31_cone_{maj,miE,miW}_mask.fits
data/cone_cutouts/hi_cube_jcomb_scaled2_submed_cone_{maj,miE,miW}_cutout.fits
data/cone_cutouts/rohsa/{maj,miE,miW}_cutout_ng3/summary_ng3/model_cube.fits
data/cone_cutouts/rohsa/{maj,miE,miW}_cutout_ng3/summary_ng3/component_0{1,2,3}_amplitude.fits
data/cone_cutouts/rohsa/{maj,miE,miW}_cutout_ng3/summary_ng3/component_0{1,2,3}_mu_kms.fits
data/cone_cutouts/rohsa/{maj,miE,miW}_cutout_ng3/summary_ng3/component_0{1,2,3}_sigma_kms.fits
```

Braces above enumerate alternatives; they are not literal filename characters.
The optional `data/rohsa_spectra_cache/` can hold the matching six-spectrum cache;
otherwise spectra are extracted from the supplied cube. This is a plotting
entry point for completed ROHSA fits, not a ROHSA installation/training pipeline.
The final overview PDF and its source-data workbook are included even when
external image products are not installed.
