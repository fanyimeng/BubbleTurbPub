# Final 118-bubble catalogue

`m31_bubble_catalog_final_yes_118.tab` is the fixed-width table used by the final
paper calculations. Read it with
`pandas.read_fwf(path, header=0, infer_nrows=1000000)`.

Use `cand_label` as the unique identifier. The historical numeric `id` is not
unique. The 118 objects, positions and native expansion velocities are retained;
the ambient ring density and derived quantities use the final global ring mask.
All 118 rows have finite ring density, momentum and supernova-count values.

| Columns | Meaning / units |
| --- | --- |
| `cand_label` | Stable `CANDxxx` identifier |
| `id` | Historical numeric catalogue identifier; may repeat |
| `ra_hms`, `dec_dms` | Equatorial position, hours/minutes/seconds and degrees/minutes/seconds |
| `ra_pix`, `dec_pix` | Pixel position in the original analysis maps; requires their original WCS |
| `major`, `minor`, `pa` | Stored ellipse axes in pixels and position angle in degrees, using the extraction-code convention |
| `maj_as`, `min_as` | Stored angular axes in arcseconds |
| `vmin_kms`, `vmax_kms`, `v1`, `v2` | Stored spectral velocity limits / measurement endpoints, km/s |
| `v_ang` | Adopted PV cut angle, degrees |
| `collide`, `good` | Historical annotation fields; not a new selection criterion |
| `radius_pc` | Adopted deconvolved bubble radius, pc |
| `expansion_vel` | Native measured expansion velocity, km/s; do not add 1 km/s |
| `r_kpc` | Stored historical galactocentric radius, kpc; retained for the calculations that explicitly use it |
| `R_center_fits_kpc` | Radius sampled at the bubble centre from the final radius map, kpc |
| `n_HI_ring_cm-3` | Final ambient H I number density from the global ring mask, cm^-3 |
| `n_HI_ring_cm-3_global_recomputed` | Same final ring density, retained as provenance |
| `n_HI_ring_cm-3_original` | Earlier ring density, including 19 missing values; not the final density |
| `n_HI_ring_was_missing`, `n_HI_ring_relative_change`, `n_HI_density_source` | Provenance of the final density replacement |
| `n_HI_cm3`, `n_HI_center_cm-3`, `n_HI_ellipse_cm-3` | Retained auxiliary density estimates, cm^-3; not substitutes for the final ring density |
| `t_exp_paper_Myr` | Expansion time `0.6 r / v_exp`, Myr |
| `dot_e_paper_1e-28` | `0.5 mu_H n_HI v_exp^2 / t_exp`, in 10^-28 erg cm^-3 s^-1 |
| `p_rad_cgs` | `(4 pi/3) r^3 mu_H n_HI v_exp`, g cm s^-1 |
| `SN_weaver` | Adopted supernova count: `1.2 (v_exp/10 km/s)^2 (r/100 pc)^3 n_HI/(1 cm^-3)` |
| `SN_num` | Retained uncorrected Weaver normalization, exactly half of `SN_weaver`; not the adopted count |

The final calculation uses `mu_H = 2.34e-24 g`, `1 pc = 3.0e18 cm` and
`1 Myr = 1e6 * 3.154e7 s`, as implemented in `plot/paper_formula_common.py`.
The density mask excludes the interiors of all final 118 bubbles before sampling
each surrounding ellipse at twice its stored axis scale.

`merged_sampling_profile_muH_2p34.tsv` contains the final turbulence rates.
`merged_sampling_profile.tsv` preserves the source normalization of 1.67e-24 g;
the final rates are scaled once by 2.34/1.67. Do not apply that correction again.
The Brinks comparison uses its own published velocity definition and the explicit
corrections in `corrected_brinks`; these do not alter the new-bubble velocities.

The separate `m31_bubble_candidates_final_yes_no_365.csv` is an existing candidate
classification archive, unchanged in this release. Use the 118-row table above
for final physical quantities and the figure scripts.
