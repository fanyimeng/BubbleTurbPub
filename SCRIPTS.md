# Script Overview (Paper Figures First)

This is a concise, paper-driven index. Figures are listed in the order they
appear in `paper/main.tex`.

## Figures Used in the Paper

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

## Key Plotting Tables (short list)

- `code/1113.tab`: main bubble catalog used by most plots; effectively also the
  ring table for `n_HI_ring_cm-3`.
- `data/profile_resampled_{max,min,med}.tsv`: resampled radial profiles used for
  turbulence and MRI curves.
- `data/profile_output_VLA_JCOMB.txt`: major-axis profile for
  `plot/hi_profile_major_axis.py`.
- `data/brinks+86/brinks86_combined.fwf`: Brinks+86 comparison table.
- `code/kang_09_table2.dat`: OB-star census table for `plot/ob_sn_density_nhi.py`.

## Other Non-figure Scripts (brief)

- `code/brinks86_combine.py` and `code/brinks86_combine_with_regions.py`:
  build the Brinks+86 combined tables used in plots.
- `data/CAR_B05_MP_C0402_1222test3_simobs_casa.py`: CASA simobserve/tclean script.
- `git_push.sh`: add/commit/push helper.
