# Historical material

The final-paper entry points are listed in [SCRIPTS.md](SCRIPTS.md). The following
material is retained for historical context, with its original conventions:

- `initial_script/`: original catalogue processing, manual GUI and PV utilities.
  See `INITIAL_SCRIPT.md`. These are not the current final-catalogue pipeline.
- `plot/v2_over_t_balance.py`, `plot/v2_over_t_vs_turb.py`, `plot/v2_vs_turb.py`,
  `plot/dot_e_vs_timescale.py` and their PDFs: earlier comparison variants.
- `plot/main_chart_sum.py`: earlier multi-panel layout, superseded by the final
  Figure 3 and Figure 4 entry points.
- `code/brinks86_combine.py`, `code/brinks86_combine_with_regions.py`: original
  preparation utilities. The final figure scripts use the included comparison
  tables and the explicit corrections in `plot/paper_formula_common.py`.
- `code/m31_bubble_candidates_final_yes_no_365.csv`: unchanged classification
  archive. Use the separate final 118-row catalogue for physical quantities.

Historical utilities may require their original environment, paths and inputs.
Their assumptions, including any older velocity or density normalization, must
not be substituted for the final equations. They are excluded from
`build_paper_figures.py` and the current environment/figure validation claims.
