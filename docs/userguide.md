# DAPPLE user guide

A walkthrough for analysts who want to load an MSI dataset, run the recommended
pipeline, browse the result, and save reproducible outputs. Written for someone
who has used napari before but is new to DAPPLE.

If you want to know **how** each step works internally — what the algorithms do,
how thresholds are picked, where the defaults come from — read the
[Algorithmic reference](algorithms.md) instead.

---

## 1. Install

DAPPLE supports Python 3.11 and 3.12. The bootstrap helpers create or reuse a
repository-local `.venv`, update packaging tools, install DAPPLE with its tested
PyQt6 binding, run `pip check`, and run the installation doctor.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

On macOS or Linux:

```bash
sh scripts/bootstrap.sh
```

These commands install the analyst/user environment. Contributors should use
`-Dev` on PowerShell or `--dev` on POSIX; that mode installs DAPPLE editable
with test, lint, typing, benchmark, and build tools.

The equivalent manual user install on Windows is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install ".[gui]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m dapple.cli.doctor
```

Use `.venv/bin/python` on macOS/Linux. The installed `dapple-doctor` command is
equivalent to the last line. It checks the Python version, analysis and CLI
imports, all four installed console launchers, Qt binding, and napari/npe2
registration plus command targets. It returns a nonzero status on a required
failure.

If `.venv` is stale or incomplete, rerun the helper with `-Recreate` on
PowerShell or `--recreate` on macOS/Linux. This explicit option removes only the
repository-local `.venv` before rebuilding it.

DAPPLE does not yet have a dependency-minimal headless extra: its base install
still includes napari, QtPy, and pyqtgraph. A CLI-only machine may omit `[gui]`
to avoid a concrete PyQt6 binding and use `dapple-doctor --headless`; that flag
makes Qt binding and widget-target failures warnings, but does not remove the
visualization dependencies.

Current processing is CPU-bound. The `cuda`, `directml`, and `mps` extras only
exercise experimental accelerator detection and do not promise faster
pipelines. Readers currently materialize arrays in RAM; lazy disk-backed
Zarr/Dask execution is not implemented.

---

## 2. Launch

Two equivalent ways to get started:

### a. Drop a dataset on the napari window

```powershell
.\.venv\Scripts\python.exe -m napari "path\to\sample.imzML"
.\.venv\Scripts\python.exe -m napari "path\to\Boone cdf"  # multi-file CDF imaging
```

DAPPLE's reader fires automatically and adds a summary projection. A
`.spec.xml` opened by itself only displays its metadata; it does not execute the
pipeline. Use `dapple-apply-spec` for an actual replay.

### b. Open the wizard, load from there

```powershell
.\.venv\Scripts\python.exe -m napari
```

Then **Plugins ▸ DAPPLE ▸ DAPPLE Wizard**. The wizard owns the loading flow and the
recommended-pipeline plumbing, and is the easiest way to get a sensible result on
a brand-new dataset.

---

## 3. Wizard walkthrough

The wizard has seven pages:

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐  ┌──────────┐  ┌────────┐  ┌──────┐
│ 1. Load  ▶│ 2. Params  ▶│ 3.Preview ▶│ 4.ROI ▶│ 5.Workflow ▶│6.Review ▶│ 7.Run │
└──────────┘  └──────────┘  └──────────┘  └─────┘  └──────────┘  └────────┘  └──────┘
```

### Page 1 — Load

Pick an imzML file, a single .cdf, or a directory of .cdf files. The wizard
detects the format automatically. After loading you'll see:

- A status line summarising pixel count, grid shape, and detected
  `instrument_family / ionization / mode / polarity`.
- A **peak count** projection layer added to the napari canvas. (Peak count is
  the default first view rather than TIC because TIC is often a constant on
  vendor-pre-normalized data — peak count varies pixel-to-pixel even then.)

### Page 2 — Experiment parameters

Every field is auto-populated from the dataset's metadata (imzML CV terms or
ANDI-MS attributes) and any sidecar file the wizard found. The colored badge
next to each row tells you the source:

| Badge color | Meaning |
| :---------- | :------ |
| **green** "auto-detected (imzML)" / "auto-detected (CDF)" | The dataset's own header gave us the value. Trust unless you have a reason not to. |
| **blue** "from sidecar metadata" / "from .spec.xml" | A `<stem>.metadata.json` / `<stem>.experiment.json` / `<stem>.spec.xml` file next to the dataset overrode the auto-detected value. |
| **purple** "user override" | You edited the field; it's no longer the detected value. |
| **amber** "default — please verify" | Neither the dataset nor a sidecar named this field; the wizard fell through to a generic default. **Read these carefully** — defaulted instrument-family / mode is the most likely cause of bad downstream defaults. |

The **Reset to auto-detected** button below the form discards every override
and restores the badged values.

> **About `sample_type`.** Review this field explicitly. `tissue` enables the
> Moran's I permutation filter in the recommended chain; dispersed samples do
> not receive that spatial-coherence assumption by default.

### Page 3 — Preview projections

Buttons for the six standard per-pixel projections:

| Projection | What it shows |
| :--------- | :------------ |
| Total ion current (TIC) | Sum of intensities. May be constant on vendor-normalized data. |
| Root-mean-square intensity | Less sensitive to a single dominant peak than TIC. |
| Median intensity | Per-pixel non-zero median. Robust; useful as a normalization reference. |
| Base peak intensity | Maximum intensity per pixel. |
| Peak count | Number of peaks per pixel. Diagnostic of detection-floor uniformity. |
| Mean m/z | Intensity-weighted mean m/z; reveals gross compositional shifts. |

Clicking any button adds (or refreshes) a napari Image layer with that
projection. Contrast is auto-set to the 1st–99th percentile of non-zero pixels,
so an empty-pixel background doesn't compress all the visible signal into a
single color.

### Page 4 — Region(s) of interest

1. Click **Add ROI layer**. A `MSI ROIs` Shapes layer is added to napari and
   polygon-drawing mode is activated.
2. Click vertices on the canvas to outline a region; press Esc to finish a
   polygon.
3. Each polygon appears as a row in the wizard's ROI table. Edit the **Name**
   inline; tick the **Background** checkbox to mark a polygon as a background
   reference rather than a structure of interest.
4. The **Treat outside-ROI pixels as background** checkbox tells the
   background-subtract operator to use everything outside any foreground polygon
   as background, even if you don't draw an explicit background polygon.

> Polygons drawn here are picked up by the **Mass Spectrum Panel** in
> polygon-aggregate mode and by the **Background Subtract** operator.

ROI names, foreground/background roles, colors, and zero-based napari `(y, x)`
vertices are saved in `.spec.xml`. Later ROI enrichment requires an explicit
overlap policy: fail on any overlap (default), exclude multiply covered pixels,
assign them to the first ROI, or allow shared membership. An enrichment
contrast always requires its numerator and denominator pixel sets to be
disjoint.

### Page 5 — Recommended workflow

The wizard reads your `ExperimentParams` and proposes a default operator chain
([details](algorithms.md#recommended-pipeline)):

1. **CWT centroiding** — first for profile data only. Dense profile samples are
   not valid reference peaks; centroiding first prevents adjacent bins from
   becoming false high-prevalence references.
2. **Reference-ion detection** — always, after profile centroiding when needed.
3. **Empirical tolerance fit** — always.
4. **Mass recalibration** — TOF / Q-TOF families only. Per-pixel piecewise-
   linear warp from observed reference m/z values onto the consensus centroids
   (with RANSAC outlier rejection). Skipped on Orbitrap / FT-ICR where the data
   is locked enough that recalibration tends to add noise.
5. **Per-pixel normalization** — median by default. ``tic_normalize`` and
   ``reference_ion_normalize`` are also offered.
6. **SNR filtering** — for centroided data only. Profile data is not CWT-picked
   a second time.
7. **Consensus peak alignment** — always.
8. **Spatial filter (Moran's I + permutation FDR)** — tissue samples only
   (``ExperimentParams.sample_type == "tissue"``). Drops consensus channels
   whose pixel-level intensity pattern is indistinguishable from random.

Each operator is a card. Hover any field to read what the parameter does, why
the default is what it is, and how to adjust it. Per-card **Reset** restores
that node's defaults; the page-level **Reset all to defaults** rebuilds every
card from scratch.

Cards for operators with multiple implementations carry a **Variant:**
dropdown in their title row so you can swap to an alternative without
dropping out of the GUI:

| Slot | Variants | When to swap |
| :--- | :------- | :----------- |
| Recalibration | ``msiwarp_recalibrate`` ↔ ``lock_mass_recalibrate`` | Lock-mass when only one trusted anchor / fewer than 3 visible anchors per pixel. |
| Normalization | ``median_normalize`` ↔ ``tic_normalize`` ↔ ``reference_ion_normalize`` | TIC only when ionization is uniform; reference-ion when matrix peaks are stable across the image. |
| Consensus | ``kde_consensus_alignment`` ↔ ``dbscan_consensus`` | DBSCAN on sparse peak pools where KDE bandwidth over-merges. |

The dropdown's tooltip describes every variant; selecting one replaces the
operator and resets its params to its own defaults (variants have different
parameter shapes — we don't try to translate field-by-field). Edits on other
cards are preserved across a swap.

The page also exposes three **optional cleanup operators**, each as a card
with its own **Enable** checkbox (default off):

- **Hot-pixel correction** — at the *top* of the chain. Replaces detector
  glitches / matrix-crystal hot pixels with their neighbors' median TIC.
  Enable when the Preview's TIC view shows isolated extreme-bright pixels that
  dominate auto-contrast.
- **Experimental prevalence sensitivity filter** — *after the recommended
  chain.* Compares channel prevalence with a with-replacement occupancy model.
  This option is disabled by default and is not a calibrated replacement for
  fixed ``min_prevalence``: consensus assignment itself constrains how many
  peaks can occupy each pixel, which can make the occupancy scores
  anti-conservative. Use it only to check sensitivity across thresholds; use
  cohort dataset prevalence when evidence across samples is available.
- **Background subtraction** — at the *bottom* of the chain. Drops consensus
  channels whose mean intensity in background pixels is comparable to or
  larger than in foreground pixels. Requires consensus alignment **and** at
  least one foreground polygon on the ROI page first.

Disabled optional cards stay visible (you can still review their parameters)
but their nodes are skipped when the Pipeline is built.

> Edits are preserved if you go **Back** to this page after running — the
> wizard tracks your edits separately from `recommend_pipeline`. After a run,
> each card shows a **rejection-budget summary** so you can see which
> parameters dropped the most candidates and which knob to tune before re-running.

### Page 6 — Review

A read-only summary of:

- The dataset (path, pixel count, grid, hash, params)
- The pipeline (every operator + every parameter value, in execution order)
- The RNG seed (deterministic re-runs)

### Page 7 — Run

Click **Run pipeline**. Execution happens on a worker thread so the UI stays
responsive; the progress bar and per-node status update live. When the run
completes:

- The post-consensus dataset is pushed back into the session, so the
  **Channels** and **Mass Spectrum** widgets see it automatically.
- A **per-node diagnostic plot strip** renders below the log: KDE density with
  consensus marks for the alignment node, an empirical-tolerance line + 95% CI
  band, per-pixel factor histogram for the normalizer, kept-peak count
  histogram for the picker, Moran's I distribution for the spatial filter, etc.
  Hover any plot's title for a tooltip explaining what it shows and which
  features mark a healthy vs. unhealthy result.
- A **diagnostic block** is appended to the run log itself with a tabular,
  health-annotated summary per operator. Each scalar key DAPPLE has a rubric
  for is graded ✓ healthy / ⚠ check / ✗ unhealthy with the threshold the
  decision was based on. Below the table, every flagged metric expands into
  a 3-line block with `what:` (what the metric measures) and `fix:` (the
  specific parameter to adjust and what to change it to). The same block
  appears in the `dapple-apply-spec` and `dapple-cohort-align` CLI output.
- A **Tune & rerun** button enables, sending you back to the workflow page so
  you can adjust parameters informed by the rejection-budget guidance now
  visible on each card.
- Three save buttons enable: `.spec.xml`, `.imzML`, and `.tif + .csv`.

If you go **Back**, edit a parameter, and return to this page without re-running:

- A `[!] Parameters changed since the last run — click Run pipeline to refresh
  results before saving.` line is appended to the log.
- The save buttons are disabled until you re-run.

The runner's in-memory cache includes operator and node id, parameter hash,
master RNG seed, input lineage, and library versions. ROI geometry is included
only for operators that declare an ROI dependency, so editing a polygon keeps
unrelated upstream numerical work reusable while invalidating background-aware
steps. The cache lasts only for that runner/session; it is not a persistent
disk cache.

---

## 4. The widgets

All widgets share state through a single `MsiSession`, so changes propagate
without any extra plumbing.

### Mass Spectrum Panel

A fast pyqtgraph plot of the spectrum at the cursor's pixel **or** aggregated
across foreground ROIs.

- **Raw / Harmonized** controls pick the data source. During a pipeline run,
  Raw is the original `PeakList` retained for the same dataset identity;
  Harmonized is the post-consensus shared-axis `PeakMatrix`. Opening a
  harmonized processed file by itself cannot recover its pre-pipeline trace, so
  only Harmonized is available. Opening an unrelated dataset clears stale raw
  state.
- **Aggregation** dropdown (mean / median / sum / max) only matters in
  polygon-aggregate mode.
- **Mode** dropdown switches between *single pixel* (cursor-driven) and
  *polygon aggregate*.
- **Log y** toggles a log-scale y-axis. Stems are filtered to non-zero
  intensities first so log(0) doesn't collapse the curve.
- **Click a stem** to surface the matching consensus channel's image — see
  *Click-to-show m/z* below.

At a single pixel, Raw plots the original native peaks. In polygon-aggregate
mode those raw spectra do not share an axis, so DAPPLE bins them onto a stable
50-ppm log-m/z grid and labels the curve **raw (binned)**. Harmonized polygon
aggregation works directly on the shared matrix. ROI edits propagate to both
views retained in the live session.

### Channels Panel

The unified per-channel viewer. One table with two sections:

- **Summary projections** (TIC, RMS, median, base peak, peak count, mean m/z) —
  always available.
- **Consensus channels** (m/z 250.1234 · prev 91%, ...) — only after consensus
  alignment.

Per-row controls:

| Column | What it does |
| :----- | :----------- |
| Show | Toggle a napari Image layer for this row. |
| Layer | Read-only label: which dataset, which projection or m/z, with prevalence. |
| LUT | Colormap for the layer. Choose a single-hue (red/green/blue/yellow/cyan/magenta) for compositing in additive mode. |
| Contrast min / max | Display contrast bounds. Auto-set on first show to 1st–99th percentile; editable thereafter. |

Page-level controls:

- **Blending** sets the blending mode for every visible layer at once
  (additive, translucent, opaque, minimum). Additive is the natural choice for
  multi-LUT overlays.
- **Show top-N by prevalence** ticks the top-N consensus rows in one click.
- **Mosaic mode** ✚ **Columns** spinbox enables napari's built-in
  `viewer.grid.enabled = True` so each visible layer renders in its own tile of
  the canvas — the easiest way to compare a handful of channels side-by-side.

### Mass Spectrum Panel — *click-to-show m/z*

Single-click on any peak in the spectrum:

1. The click position is snapped to the nearest plotted m/z (cyan vertical
   marker).
2. The session's `show_mz_requested` signal fires.
3. The Channels Panel finds the consensus channel with the closest m/z, ticks
   its **Show** box, scrolls the table to it, and selects the row. The
   corresponding image lights up on the canvas.

This is the fastest way to drill from spectrum → image.

### MSI Preview

A standalone version of the wizard's *Preview projections* page — useful when
you've loaded a dataset directly (without the wizard) and want quick access to
the projection buttons.

### ROI Widget

The wizard's *ROI* page exposed as a standalone widget. Useful when you opened
a dataset directly and want polygon ROIs without going through the full wizard.

### Spatial and developmental patterns

Open **Plugins ▸ DAPPLE ▸ Spatial and developmental patterns** after consensus
alignment. The panel is disabled for a raw `PeakList` because comparisons need
the shared channels of a harmonized `PeakMatrix`.

- **ROI enrichment** selects one foreground ROI as numerator and another as
  denominator, with `error`, `exclude`, or `first` overlap handling, trim
  fraction, and minimum pixels/group. The result table ranks m/z channels by
  q-value/effect and reports log2 fold change, q-value, and prevalence
  difference; the plot is an effect-versus-significance view.
- **Developmental axis** accepts zero-based start/end `(y, x)`, endpoint labels,
  an optional corridor half-width, optional single-ROI restriction, bins,
  permutations, and seed. You may enter coordinates or select a napari Shapes
  line and click **Use endpoints of active Shapes line**. The table reports
  pattern, rho, q-value, and end/start enrichment; selecting a row plots that
  channel's binned profile.
- **Show selected ion** sends the table row's m/z to Channels. Double-clicking a
  row does the same. **Export tables…** writes the same CSV/JSON result contract
  used by the CLI.

The panel repeats the statistical warning deliberately: ROI q-values are a
pixel-level within-image screen, not biological-replicate evidence.

### Threshold Explorer

A what-if view onto the most recent run's rejection budget, without re-running
anything.

After a successful pipeline run, the consensus alignment node records every
pre-filter candidate's prevalence and KDE prominence (peak picking and
reference detection record analogous arrays). The Threshold Explorer lets you:

- pick a filter (consensus prevalence, consensus prominence) from the dropdown,
- view the empirical CDF of candidate values,
- drag the red vertical line to preview a different threshold,
- read a live "X of Y candidates would survive at threshold Z" label that
  also reports the change vs. the operator's actual current threshold.

For prevalence, the slider value maps directly to `min_prevalence`. For KDE
prominence, the slider is a **density cutoff** but the workflow field is a
**density quantile**; use the plot to decide whether to raise or lower
`min_prominence_quantile`, not to copy the displayed density number. Then click
Run pipeline. The runner short-circuits every node whose parameters did not
change, so a one-knob re-run is fast.

The panel auto-discovers the active wizard via the napari viewer's docked
widgets; opening it as a standalone widget after a wizard run is a one-click
operation.

### Cohort harmonization

Single-dataset wizards harmonize one image. The **Cohort harmonization** panel
(under `Plugins → DAPPLE → Cohort harmonization`) does the same thing across a
*directory* of imzML files — every dataset ends up on a shared m/z axis so they
can be co-clustered or stacked.

The flow:

1. Pick a cohort root directory (or browse).
2. Optionally tweak the file pattern (default `*.imzML`), recursive toggle,
   bandwidth, per-dataset recalibration/normalization, pool weighting, and the
   prevalence denominator.
3. **Discover datasets** — pre-flight: lists every matching file plus its
   pixel count without running. Use this to confirm the cohort before
   committing to a long run.
4. **Run cohort alignment** — runs on a worker thread; the log streams
   progress and the diagnostic block at the bottom shows the same
   health-rubric output you see at the end of a wizard run, scoped to the
   cohort (`shared consensus channels`, `cohort prevalence median`, etc.).

Outputs land in the chosen output directory (defaults to
`<cohort_root>/dapple_cohort/`):

- `<stem>_cohort.imzML + .ibd + .dapple-axis.json` per dataset (keep all three together)
- `<stem>_cohort.tif + _channels.csv` per dataset
- `cohort_summary.json` — `shared_consensus_mz`, pixel-weighted
  `cohort_prevalence`, one-vote-per-dataset `dataset_prevalence`, per-dataset
  prevalence vectors, selected filter, parameters, identities, and diagnostics

Each per-dataset `.imzML` is a DAPPLE-harmonized export — open it later in
napari and it loads back into the Channels Panel and Spectrum Panel as a
PeakMatrix-backed dataset (consensus rows, "harmonized" toggle, etc.) without
having to re-run any pipeline.

The default **Equal weight per dataset** pool normalizes KDE weight within each
dataset, preventing a large or high-intensity acquisition from determining the
shared peaks. **Raw pooled intensity** deliberately restores that dominance.
The prevalence denominator is independent: **All cohort pixels** answers how
common a channel is over measured pixels and weights large images more;
**Datasets carrying the channel** asks whether each dataset has at least one
carrier and gives every dataset one vote. Both vectors are exported regardless
of which one enforces the minimum.

---

## 5. Common workflows

### Drop-and-go (no wizard)

1. `.\.venv\Scripts\python.exe -m napari path\to\dataset.imzML` (or directory).
2. Open **Plugins ▸ DAPPLE ▸ Channels** — you'll see the summary projections.
3. Open **Plugins ▸ DAPPLE ▸ Mass Spectrum Panel** — hover the canvas to see
   per-pixel spectra.
4. Click **Plugins ▸ DAPPLE ▸ DAPPLE Wizard** any time you want to run the
   pipeline; the wizard picks up the already-loaded dataset.

### Re-applying a saved pipeline

Opening `dataset.spec.xml` in napari shows a metadata summary only. To actually
restore the operator chain and run it on a fresh input, use
`dapple-apply-spec` (documented below). Serialized ROI polygons are restored
automatically only when the input hash matches. On a different input,
`--reuse-rois` is an explicit assertion that the image has already been
registered to the same pixel coordinate system.

### Sidecar metadata

Drop a `<stem>.metadata.json` next to the dataset to override auto-detected
parameters:

```json
{
  "instrument_family": "tof_reflectron",
  "ionization": "maldi",
  "polarity": "negative",
  "sample_type": "tissue",
  "notes": "DHB matrix, negative mode, 50 µm raster"
}
```

When the wizard loads, the corresponding ParamsPage rows turn blue
("from sidecar metadata") and a banner above the form points at the file. Any
keys not in `ExperimentParams` are silently ignored, so it's safe to add
project-local fields.

### Hot-pixel correction

Hot pixels are detector glitches that show up as one bright cell against the
tissue map. To detect and replace them:

1. Enable the **Hot-pixel correction** optional card at the top of the workflow.
2. Pick a correction strategy: `neighbors_median` (preserves peak count by
   scaling the spike to its neighbors' median TIC) is the conservative default.
3. The diagnostic tells you how many pixels were flagged at your chosen `k_mad`
   threshold.

### Background subtraction

Once you've run consensus alignment **and** drawn at least one foreground (and
optionally one background) ROI:

1. Enable the **Background subtraction** optional card after consensus.
2. Mode `reject_channels` (the default) drops any consensus channel whose
   `mean(bg) / mean(fg) ≥ 0.5`. Lower the threshold to be stricter.
3. Mode `subtract` instead removes the per-channel background mean from every
   pixel and clips at zero — only safe when the background is uniform.

If you tick **Treat outside-ROI pixels as background** on the wizard's RoiPage,
the operator falls back to "everything outside the foreground" as background
when no explicit bg polygon is drawn.

### Lock-mass recalibration (single-anchor shift)

If your dataset has **only one** reliable reference ion per pixel — a
spiked-in standard, a dominant matrix peak, or any case where the
``msiwarp_recalibrate`` linear-fit RANSAC needs ≥3 anchors and you only have
one or two — use ``lock_mass_recalibrate`` instead:

1. Build the recommended pipeline.
2. Edit the chain to swap out ``msiwarp_recalibrate`` for ``lock_mass_recalibrate``.
3. ``anchor_strategy`` defaults to ``"highest_intensity"`` (the most defensible
   choice — high-intensity anchors carry less m/z uncertainty).
4. If you want to lock to a *specific* m/z (a known stable internal standard),
   set ``anchor_strategy = "closest_to_mz"`` and ``explicit_anchor_mz = <m/z>``.
5. ``max_shift_ppm`` (default 500) bounds the per-pixel shift the operator is
   willing to apply — pixels whose anchor sits more than this far from the
   reference centroid are left untouched, since a giant shift on one anchor is
   more likely a misidentification than a real drift. Tighten for high-resolution
   instruments; relax for axial linear MALDI-TOF where 500 ppm of drift across
   the raster is plausible.

The operator's diagnostic reports `fraction_pixels_recalibrated` and the
distribution of `|ppm shift|`. Aim for a fraction near 1 (most pixels found a
visible anchor) and a median shift well below `max_shift_ppm`.

### DBSCAN consensus alignment (alternative to KDE)

If KDE consensus is over-merging your peaks (one consensus channel sitting
between two real peaks) or under-detecting them (real peaks below the
prominence threshold), try ``dbscan_consensus`` as a drop-in replacement:

1. Edit the pipeline's ``kde_consensus_alignment`` node and replace it with
   ``dbscan_consensus``.
2. Default ``eps_ppm = 0`` means "use 2× the upstream tolerance curve median"
   — the right choice when you trust the empirical-tolerance fit. Set a fixed
   ppm (e.g. 50) if you'd rather bypass the curve.
3. ``min_samples = 5`` is DBSCAN's core-point threshold. Drop to 3 on very
   sparse pools; raise on dense data where 5 isn't enough to silence noise.

DBSCAN gives a hard "this peak ↦ that cluster" assignment, where KDE smooths
peaks into a continuous density. On dense, well-resolved data they agree to
within a few channels; on sparse pools or unusually close peak pairs, DBSCAN's
discrete clustering tends to behave better.

### Cohort harmonization

To compare or co-cluster multiple imzML files on a single shared m/z axis, use
the cohort-align flow. Your data should already be on disk as a directory of
``.imzML + .ibd`` pairs:

```
cohort_root/
  control_run1.imzML
  control_run1.ibd
  control_run2.imzML
  control_run2.ibd
  treated_run1.imzML
  treated_run1.ibd
  ...
```

The simplest invocation is the command-line workflow, which does not open a
napari window:

```powershell
.\.venv\Scripts\python.exe -m dapple.cli.cohort_align `
  "C:\path\to\cohort_root" -o "C:\path\to\cohort_aligned"
```

Outputs in `cohort_aligned/`:

- `<stem>_cohort.imzML + .ibd + .dapple-axis.json` for each input — same m/z
  axis, ready to load; keep the three files together.
- `<stem>_cohort.tif + _cohort_channels.csv` for each input — multipage TIFF
  on the shared axis, sidecar CSV with per-channel `mz` / `prevalence`.
- `cohort_summary.json` — `shared_consensus_mz`, pixel- and dataset-prevalence,
  per-dataset prevalence, selected filter, parameters, input identities, and
  run diagnostics.

Common flags:

```powershell
# Sample-balanced peak discovery (default) and one presence vote per dataset
.\.venv\Scripts\python.exe -m dapple.cli.cohort_align cohort_root `
  --pool-weighting sample --prevalence-basis dataset `
  --min-prevalence 0.5 -o out

# Skip per-dataset MSIWarp recalibration (already-recalibrated data)
.\.venv\Scripts\python.exe -m dapple.cli.cohort_align cohort_root `
  --no-recalibrate -o out

# Walk subdirectories
.\.venv\Scripts\python.exe -m dapple.cli.cohort_align cohort_root `
  --recursive -o out

# Deliberately let raw pooled intensity drive shared-peak discovery
.\.venv\Scripts\python.exe -m dapple.cli.cohort_align cohort_root `
  --pool-weighting intensity -o out
```

The default sample-balanced pool gives every dataset equal total KDE weight.
`--pool-weighting intensity` lets larger/brighter datasets dominate candidate
discovery. Separately, `--prevalence-basis pixel` (default) thresholds the
fraction of all cohort pixels carrying a channel, whereas `dataset` thresholds
the fraction of datasets with at least one carrier. The first weights large
images more; the second is a presence/absence vote, not an abundance test.

Every output has the same channels in the same order, but raster shapes can
differ. Register/pad anatomy explicitly before stacking spatial tensors.

> The cohort flow is also exposed in code: ``from dapple.cohort.align import
> align_cohort, load_cohort_directory``. Use this when you want to inject the
> result into a downstream Python analysis without going through the JSON
> manifest.

### ROI enrichment and developmental-axis patterns

These analyses require a harmonized DAPPLE imzML. Keep three files together:
`.imzML`, `.ibd`, and `.dapple-axis.json`; the sidecar restores the shared
`PeakMatrix`. ROI polygons live separately in the run's `.spec.xml`, including
their names, foreground/background roles, colors, and zero-based `(y, x)`
vertices.

Compare one ROI (or repeated-name union) with another:

```powershell
.\.venv\Scripts\python.exe -m dapple.cli.analyze_patterns roi `
  out\embryo_harmonized.imzML `
  --roi-spec out\embryo_harmonized.spec.xml `
  --numerator head --denominator trunk `
  --overlap-policy exclude -o out\patterns --stem head_vs_trunk
```

Positive `log2_fold_change` means enriched in the numerator. Repeat
`--numerator` or `--denominator` to form a named ROI union. The overlap choices
are `error` (default), `exclude`, `first`, and `allow`; even with `allow`, the
final numerator and denominator masks must not share pixels.

Analyze a directed anterior-to-posterior segment, optionally restricted to a
saved ROI union:

```powershell
.\.venv\Scripts\python.exe -m dapple.cli.analyze_patterns axis `
  out\embryo_harmonized.imzML `
  --start 40 15 --end 40 180 `
  --axis-name AP --start-label anterior --end-label posterior `
  --half-width 25 --bins 20 --permutations 999 --seed 0 `
  --roi-spec out\embryo_harmonized.spec.xml --roi-name embryo `
  -o out\patterns --stem ap_axis
```

Axis coordinates follow napari image order **Y X**, zero-based. `t=0` is the
named start and `t=1` the named end. The endpoint effect is end over start;
reversing the endpoints negates Spearman rho and endpoint log2 enrichment and
maps peak position to `1-t`, while the two-sided permutation p/q values and
concentration remain unchanged for the same seed.

The equivalent core API is:

```python
from dapple.analysis import (
    DirectedAxis,
    analyze_axis_profiles,
    analyze_roi_enrichment,
    export_analysis_result,
    rasterize_rois,
)
```

`rasterize_rois` creates explicit populated-pixel masks;
`analyze_roi_enrichment` and `analyze_axis_profiles` return immutable result
objects with pandas table helpers; `export_analysis_result` writes CSV tables
and a JSON manifest containing the source-data hash, orientation/axis geometry,
scientific thresholds, software versions, inference status, warnings, and a
spatial fingerprint. See the [analysis algorithms](algorithms.md#roi-enrichment-analysis)
for signatures and label precedence.

> **Statistical interpretation:** ROI Welch tests use pixels as units. They are
> within-image screens, not biological-replicate tests, and spatial
> autocorrelation can make their p-values optimistic. Axis bin permutations
> likewise test ordered structure inside one image. For population claims,
> aggregate ROI effects or axis profiles within each independent specimen and
> fit the biological model with specimens—not pixels—as independent units.

### Command-line reapplication via `dapple-apply-spec`

To re-run a saved pipeline on a new dataset without launching napari:

```powershell
# Apply spec to fresh dataset, write all three outputs
.\.venv\Scripts\python.exe -m dapple.cli.apply_spec `
  "C:\path\to\new_dataset.imzML" "C:\path\to\saved.spec.xml"

# Choose where outputs go
.\.venv\Scripts\python.exe -m dapple.cli.apply_spec `
  dataset.imzML saved.spec.xml -o out\dataset_harmonized

# TIFF only (skip imzML write)
.\.venv\Scripts\python.exe -m dapple.cli.apply_spec `
  dataset.imzML saved.spec.xml --no-imzml -o out\

# Override the spec's RNG seed (for reproducibility experiments)
.\.venv\Scripts\python.exe -m dapple.cli.apply_spec `
  dataset.imzML saved.spec.xml --rng-seed 42 -o out\
```

The CLI:

1. Loads the spec; reconstructs the `Pipeline` (operator chain + params + RNG
   seed) from XML.
2. Loads the input dataset (auto-detects imzML, single-CDF, or multi-CDF
   directory).
3. Runs the pipeline through the same `PipelineRunner` the wizard uses. The CLI
   process starts with an empty in-memory cache.
4. Writes the post-pipeline dataset out as imzML, multipage TIFF, and a fresh
   `.spec.xml` capturing the rerun.

This is the right tool for batch processing (a directory of new acquisitions
all run through the same SOP), CI-style lineage checks, and non-interactive
server use cases (with the base visualization dependencies still installed).
Numerical regression checks should pin the environment and compare
scientific outputs with declared tolerances; a saved spec does not guarantee
bit-identical results across libraries or platforms.

---

## 6. Outputs

Three formats from the **Run** page after a successful run:

### `.spec.xml`

A small XML file capturing every operator + parameter + RNG seed + library
versions, a content-addressed input hash, scalar diagnostic summaries, and saved
ROI definitions. `dapple-apply-spec` restores ROI geometry on a hash-matched
input; reusing pixel coordinates on a registered but different input requires
`--reuse-rois`. Array diagnostics such as bootstrap envelopes and KDE
curves are not serialized automatically; there is no sibling Zarr diagnostic
store in the current implementation.

### `.imzML` (round-trip)

A processed-mode imzML + paired `.ibd` of the harmonized dataset. The MD5
declared in the imzML is recomputed at write time and matches the `.ibd`. Other
tools (Cardinal, METASPACE, etc.) can read it directly.

When the source dataset has a post-consensus `PeakMatrix` backend, DAPPLE
also writes a sibling `<base>.dapple-axis.json` carrying the shared m/z axis
plus per-channel prevalence, and embeds a `dapple-harmonized` user-param
marker in the imzML XML. **On reload, DAPPLE detects the marker and
reconstructs the dense matrix automatically** — the Channels Panel shows the
consensus channel rows, the Mass Spectrum Panel's *Harmonized* toggle works,
and the Threshold Explorer has data to operate on. Round-tripping a harmonized
dataset and continuing to work on it is fully supported.

The sidecar does not contain the original pre-pipeline peak lists. If you open a
harmonized imzML in a new session, DAPPLE can show the harmonized matrix but not
a Raw comparison. Raw-vs-harmonized overlay is available when the pipeline ran
from the original PeakList in the current session.

Third-party tools that don't know about the marker / sidecar see a normal
processed-mode imzML and read it as a per-pixel sparse peak list, which is the
correct fallback.

### Multipage `.tif` + `_channels.csv`

- The `.tif` is a BigTIFF with one float32 page per consensus channel,
  `photometric='minisblack'`, optional deflate compression.
- Each page's `description` tag carries a JSON blob with the channel's m/z,
  prevalence, and an `op_history_hash` linking it to the producing pipeline.
- The sidecar `<base>_channels.csv` repeats the per-page metadata in
  human-readable form: `page, mz, prevalence` plus any extras you passed.

The TIFF can be opened directly in ImageJ/Fiji; the CSV gives you a quick
manifest in Excel or pandas.

### Pattern-analysis tables

ROI analysis writes `<stem>_roi_enrichment.csv`; axis analysis writes
`<stem>_axis_profiles.csv` and `<stem>_axis_statistics.csv`. Both also write
`<stem>_manifest.json` with schema version, named contrast or axis/endpoint
labels, source-data hash, exact directed-axis coordinates/corridor, scientific
thresholds, software versions, inference status, warnings, table names, and a
spatial fingerprint. The fingerprint covers the analysis-visible mask or
directed geometry. Archive the harmonized imzML sidecar and `.spec.xml` beside
the tables as well: the manifest verifies the source state and mask selection
but does not embed the original spectra, pipeline, or ROI polygons.

---

## 7. Troubleshooting

### "DAPPLE Wizard" not in the Plugins menu

```powershell
.\.venv\Scripts\python.exe -m dapple.cli.doctor
```

Repair any required failure it reports. Contributors can rerun the bootstrap
with `-Dev`; users can rerun the normal bootstrap. The doctor verifies the same
npe2 discovery path napari uses.

### TIC projection looks bimodal

Most likely the data is vendor-pre-normalized (every pixel sums to the same
TIC). Use **peak count** or **median intensity** instead — both vary
meaningfully even on pre-normalized data.

### "Draw at least one foreground ROI" in polygon-aggregate mode

Confirm a `MSI ROIs` Shapes layer exists in napari and contains at least one
polygon. The Mass Spectrum Panel auto-syncs from any Shapes layer in the
viewer; if the panel still doesn't pick up your polygons, drag a vertex on the
canvas to nudge a refresh.

### Pipeline run says "FAILED: No consensus peaks survived..."

Your `min_prevalence` threshold is stricter than the data supports. The
diagnostic prints the largest prevalence found; on the wizard's WorkflowPage,
edit the **Drop peaks present in < this fraction of pixels** field on the
Consensus Peak Alignment card to a value below that maximum.

### Save buttons are disabled after going Back to Workflow

You changed a parameter; the cached results are stale. Click **Run pipeline**
again (the runner caches unchanged nodes, so this is fast).

### Log y mode hides the spectrum

Already fixed — stems are filtered to `y > 0` and use a finite baseline in log
mode. If you still see this, ensure you're on the latest commit.

---

## 8. Where to go next

- **[Algorithmic reference](algorithms.md)** — what each operator computes,
  parameter semantics, and the rationale behind every default.
- **`pipeline/recommend.py`** — read this to understand which operators land
  in the default chain for your instrument family.
- **`tests/test_real_jerboa.py` / `tests/test_io_cdf_image.py`** — minimal
  end-to-end recipes against the supplied real datasets; useful as scripting
  templates if you want to drive DAPPLE from your own Python code rather than
  through the wizard.
