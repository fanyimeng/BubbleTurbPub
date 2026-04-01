# initial_script Overview

This folder contains early and ad-hoc scripts used to build the M31 HI bubble
catalog, extract subcubes, and generate PV plots. Most scripts are small
drivers that call the core library in `bubturb.py`.

## Shared Concepts and Data Formats

- Fixed-width tables (`.tab`): Most scripts read with `pd.read_fwf(...)` and
  write with `tabulate` or `DataFrame.to_string`. Columns are often numeric and
  should preserve ordering for downstream tools.
- DS9 region files (`.reg`): Ellipse definitions used for visualization and
  cross-checking. These are read/parsed or generated from tables.
- Spectral cube FITS: The main HI data cube and derived subcubes are read via
  `spectral_cube.SpectralCube` and plotted with `pvextractor`.

## File-by-File Notes

### `initial_script/0407-sorted.tab`

Fixed-width table of bubble candidates after sorting/reindexing. Header shows
the columns:

- `id`: Sequential ID (often reassigned by declination sort).
- `ra_pix`, `dec_pix`: Pixel center in the FITS image.
- `major`, `minor`: Ellipse axes in pixel units (from DS9).
- `pa`: Position angle (deg).
- `ra_hms`, `dec_dms`: Sky coordinate strings (J2000).
- `maj_as`, `min_as`: Ellipse axes in arcsec (often from pixel scale * factor).
- `vmin_kms`, `vmax_kms`: Velocity range (km/s) for the bubble.
- `collide`: Flag for collision/overlap (0/1).
- `v1`, `v2`: Manual velocity picks (km/s).
- `v_ang`: Angle for the PV slice that best shows the bubble.
- `good`: Quality flag (e.g., `pp`, `vv`, `pv`, `nn`).
- `date`: Last edited timestamp (YYMMDD-HHMM).

This file is used as the main catalog input for plotting and Streamlit review.

### `initial_script/bubturb.py`

Core library with region parsing, geometry helpers, subcube extraction, PV
plotting, and table transformations. It is the main dependency for almost all
other scripts in this folder.

#### Region Parsing and Attributes

- `read_reg_file(file_path)`: Returns full text of a DS9 region file.
- `parse_global_properties(global_line)`: Parses DS9 `global` properties into a
  dict via regex (key=value or key="value").
- `parse_region_attributes(attributes_str)`: Parses DS9 region attributes after
  the `#` sign. If `text={-v1-v2}` is found, it extracts and sorts velocities.
- `parse_ellipse_regions(reg_content)`: Parses a region file into a dict with a
  `global` section and a list of ellipses. Ensures `major >= minor` and adjusts
  `pa` when axes are swapped; assigns a running `id`.
- `read_ds9_ellipse_regions(reg_path)`: Reads a DS9 file and returns ellipse
  params in sky coordinates (as strings). Also extracts the text label.
- `read_ds9_reg_file_to_df(filename)`: Reads DS9 ellipse lines into a list of
  dicts with `coorx`, `coory`, `major`, `minor`, `pa` (pixel coords).

#### Geometry and Coordinate Transforms

- `calculate_relative_endpoints(center_x, center_y, angle, length)`: Returns
  two endpoints for a line segment of given length at a given angle.
- `create_pv_path(endpoints)`: Creates a `pvextractor.Path` from endpoints.
- `celestial_to_pixel(ellipse_params, wcs)`: Converts ellipse center/axes from
  RA/Dec + arcsec to pixel coordinates (uses a fixed `major/5` conversion and
  scales axes by 1.2).
- `is_point_in_ellipse(x, y, ellipse)`: Ellipse inclusion test in pixel space.
- `ellipse_line_intersection_improved(...)`: Uses `scipy.optimize.fsolve` to
  intersect a line (at `pa_seg`) with an ellipse.
- `max_min_values_in_ellipse(cube, ellipse, wcs)`: Computes min/max intensity
  within an ellipse on a moment-0 map.

#### Subcube Extraction

- `export_subcube_v2(cube, ellipse, output_file=None, scale_factor=4,
  v_plot_range=40)`: Extracts a spatial subcube around the ellipse center. If
  `major < 80`, expands the extraction size. Velocity slice is
  `[vmin - v_plot_range, vmax + v_plot_range]`. Writes FITS if `output_file`.
- `export_subcube_v3(cube, ellipse, output_file=None, scale_factor=4,
  v_plot_factor=1.5)`: Similar to v2 but uses WCS pixel scales to compute pixel
  sizes and expands velocity range by a *factor* around the midpoint.

#### PV Plot Pipelines (v4 to v9)

These functions generate multi-panel figures: moment-0 maps plus six PV slices
at angles `theta + n*30 deg`. All versions rely on the same inputs:

- `cube` (SpectralCube), `theta` (base angle), `ellipse` dict
  (`major`, `minor`, `pa`, `velocities`, `v_ang`, `v1`, `v2`).
- Optional `reg_file` + `plot_extra_ellipses` to overlay B86 ellipses.
- Most versions draw:
  - A central moment-0 map (subplot 1).
  - A velocity-range moment-0 map (subplot 5).
  - Six PV diagrams (other subplots), with guide lines for spatial/velocity
    extent and for the `v_ang` direction.

Version notes:

- `bubble_pv_plot_v4(...)`: Baseline layout; draws 6 PV slices, overlays ellipse,
  and computes a rough mass/energy estimate by summing pixels inside the
  velocity-range moment-0 map.
- `bubble_pv_plot_v5(...)`: Adds DS9 ellipse overlay support and improves label
  rendering; mass/energy computation is commented out and returns `0`.
- `bubble_pv_plot_v6(...)`: Reframes moment-0 plots with tighter spatial bounds,
  sets aspect to `auto`, and computes pixel scale from WCS for more consistent
  sizing.
- `bubble_pv_plot_v7(...)`: Adds clearer angle path annotations and optional
  overlay ellipses; focuses on visualization consistency.
- `bubble_pv_plot_v8(...)`: Adds basic parallelization (ThreadPoolExecutor)
  for moment-0 generation and PV extraction; same plotting logic.
- `bubble_pv_plot_v9(...)`: Uses parallel PV extraction and normalizes each PV
  panel using only the target velocity range for contrast.

#### Table and Region Conversion Utilities

- `generate_synthetic_ellipses_dataframe(real_ellipse_dict, wcs)`: Converts
  DS9 ellipse pixel coords to a DataFrame with pixel positions and axes.
- `save_dataframe_to_fixed_width(df, filename)`: Writes fixed-width text via
  `df.to_string`.
- `reg2df(fits_image_path, ds9_reg_file_path, output_file_path)`: DS9 -> table
  using WCS conversion; writes fixed-width table.
- `normalize_ellipses(df_path, output_file_path)`: Swaps major/minor when needed
  and normalizes `pa` to `[0, 180)`.
- `add_ra_dec_strings(df_path, fits_path, output_file_path)`: Adds `ra_hms` and
  `dec_dms` columns from pixel coords using WCS.
- `major_minor_2arcsec(df_path, output_file_path, factor=1)`: Adds `maj_as`,
  `min_as` columns by multiplying pixel axes by `factor`.
- `pandas2ds9(ascii_file_path, ds9_file_path)`: Converts a tab file with
  `ra_hms`, `dec_dms`, `maj_as`, `min_as`, `pa` to a DS9 region file.

#### Catalog Editing and Derived Quantities

- `reassign_ids_by_dec(df_path, output_file_path, id_col='id', sort_col='dec_pix')`:
  Sorts by a column and reassigns a sequential `id`.
- `selectAccordingToColumnValue(...)`: Filters rows by a column/value.
- `compute_center_velocity(df_path, output_file_path)`: Adds `center_vel` as
  mean of `v1` and `v2`.
- `compute_expansion_velocity(df_path, output_file_path)`: Adds `expansion_vel`
  as `|v2 - v1|/2`.
- `compute_radius_pc(df_path, output_file_path)`: Computes `radius_pc` from
  `maj_as` via `maj_as / 60 * 216`.
- `compute_radius_pc_deconvolve(df_path, output_file_path, beamsize=216.)`:
  Deconvolved radius using `sqrt((maj_as/60*216)^2 - beamsize^2)`.

#### B86 HI Hole Table Conversion

- `parse_mixed_format_file(input_path)`: Parses a pipe-delimited table with
  B1950 RA/Dec split fields and returns a DataFrame.
- `process_hi_hole_with_ellipses(...)`: Converts B1950 RA/Dec to J2000, writes
  a fixed-width table, and generates a DS9 ellipse region file where major/minor
  are converted from pc to arcsec using `pc_per_arcsec`.

### `initial_script/app_v7.py`

Streamlit UI for interactive review and editing of bubble parameters.

Core behavior:

- Loads `0407-sorted.tab` into a DataFrame, ensures `good` and `date` columns
  exist, normalizes `collide` and `in_b86` to int.
- Uses `SpectralCube` to load each subcube by ID.
- Shows a left panel with editable parameters (`ra_hms`, `dec_dms`, `maj_as`,
  `min_as`, `pa`, `vmin_kms`, `vmax_kms`, `v1`, `v2`, `v_ang`, `collide`, `in_b86`).
- Saves edits back to `0407-sorted.tab` and marks `good` status using buttons.
- Right panel displays PV plot image (`pvplots/temp.png`) and provides controls
  to replot, save subcube, navigate IDs, and load existing plots.

Hard-coded paths to subcubes and data are set near the top:
`SUBCUBE_PATH`, `ORIGINAL_SUBCUBE_PATH`, `PVPLOT_DIR`, and `FITS_PATH`.

### `initial_script/cc00_dfProcess.py`

Pipeline driver for table conversions. The active commands currently:

- `reassign_ids_by_dec('0328-final_with_collision_fields.tab', '0407-sorted.tab')`
- `reassign_ids_by_dec('0407-sorted.tab', '0407-sorted.tab')` (re-run for order)

Commented steps show a typical pipeline:

1. `reg2df(...)` DS9 to fixed-width table.
2. `normalize_ellipses(...)` fix major/minor and PA.
3. `add_ra_dec_strings(...)` add sky coordinates.
4. `major_minor_2arcsec(...)` convert axes to arcsec.
5. `pandas2ds9(...)` regenerate DS9 regions.
6. `process_hi_hole_with_ellipses(...)` convert B86 catalog to J2000 + regions.

### `initial_script/cc01_subplot.py`

Single-process batch PV plotting:

- Reads `0406-sorted.tab`.
- Loads the full HI cube from `../data/jcomb_vcube_scaled2.fits`.
- For each ID, extracts a subcube with `export_subcube_v2` and plots PV figures
  via `bubble_pv_plot_v5`.
- Outputs to `/Users/meng/alex/astro/m31/00bubble/subcube_0406` and saves plots
  under `t1+t2/`.

### `initial_script/cc01_subplot_MP.py`

Multiprocessing version of the PV plotting pipeline:

- Reads `0407-sorted.tab`.
- Uses `ProcessPoolExecutor` to process all IDs in parallel.
- Each worker loads the cube, extracts a subcube (v3), writes FITS, then creates
  a PV PNG in `./pvplots/`.
- Uses `os.cpu_count()` as `max_workers`.

### `initial_script/cc02_select_good.py`

Small driver to reassign IDs or filter by quality flag.
Currently active: `reassign_ids_by_dec('0604-gtp.tab', '0604-gstp.tab')`.

### `initial_script/CC03_trivial_parameters.py`

Adds derived kinematic/size columns to a catalog:

- `reassign_ids_by_dec(...)`
- `compute_center_velocity(...)`
- `compute_expansion_velocity(...)`
- `compute_radius_pc(...)`

### `initial_script/CC04_sort.py`

Reassigns IDs after sorting by a column. The active line sorts by `r_kpc`:

```
reassign_ids_by_dec(df_path='0705.tab', output_file_path='0705-s.tab',
                    sort_col='r_kpc')
```

### `initial_script/CC05_size_deconvolve.py`

Deconvolves beam size from the physical radius:

- Example calls for `compute_radius_pc_deconvolve(...)`.
- Active line uses `beamsize=0.` on `0709.tab` to produce
  `0709-decon_hb.tab`.

### `initial_script/ds9_2_pandas.py`

Standalone conversion from DS9 region file to fixed-width table:

- Reads FITS WCS (`../data/jcomb_vcube_scaled2_mom0.fits`).
- Converts DS9 pixel ellipses to sky coordinates.
- Writes `0328-final_xy.tab` with many placeholder columns (velocity, mass,
  energy, etc.) set to NaN for later editing.

### `initial_script/pandas_2_ds9.py`

Standalone conversion from a fixed-width table back to DS9 regions:

- Reads `0709.tab` and writes `0709.reg`.
- Uses `ra_hms`, `dec_dms`, `maj_as`, `min_as`, `pa`, and `id`.

### `initial_script/SFR_Kang.py`

Computes OB mass formation rates from Kang+ (2009) Table 2:

- Fixed-width reader with explicit `colspecs`.
- Converts sentinel values (-99, -9.9e+01) to NaN.
- Bins by user-defined `AGE_BINS` (Myr), sums `M_02`, and converts to
  mass formation rate per bin (Msun/yr).
- Prints a summary and writes `ob_age_mass_rate.csv`.

## Typical Data Flow (Example)

1. DS9 region file -> `reg2df` / `ds9_2_pandas.py` to create a fixed-width table.
2. Normalize ellipse params, add RA/Dec strings, convert axes to arcsec.
3. Reassign IDs and curate the table (`cc00_dfProcess.py`).
4. Extract subcubes and generate PV plots (`cc01_subplot_MP.py`).
5. Review and edit in `app_v7.py`, set `good` flags and velocity picks.
6. Filter/select high-quality entries and compute derived parameters
   (`cc02_select_good.py`, `CC03_trivial_parameters.py`).
