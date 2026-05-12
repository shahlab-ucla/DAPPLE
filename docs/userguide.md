# DAPPLE user guide

A walkthrough for analysts who want to load an MSI dataset, run the recommended
pipeline, browse the result, and save reproducible outputs. Written for someone
who has used napari before but is new to DAPPLE.

If you want to know **how** each step works internally — what the algorithms do,
how thresholds are picked, where the defaults come from — read the
[Algorithmic reference](algorithms.md) instead.

---

## 1. Install

DAPPLE is a normal Python package; you install it into a virtual environment and
run `napari` to pick it up automatically.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

pip install -e ".[dev]"           # editable install with test deps
napari
```

Confirm DAPPLE is registered:

```powershell
python -c "from npe2 import PluginManager; pm = PluginManager.instance(); pm.discover(); print(pm.get_manifest('dapple'))"
```

You should see `dapple (DAPPLE)` listed with three readers and five widgets.

---

## 2. Launch

Two equivalent ways to get started:

### a. Drop a dataset on the napari window

```powershell
napari "path\to\sample.imzML"
napari "path\to\Boone cdf"          # directory of multi-file CDF imaging
napari "path\to\sample.spec.xml"    # re-applies a saved pipeline
```

DAPPLE's reader fires automatically. A summary projection layer (peak count by
default) is added to the canvas and the plugin's widgets become available under
**Plugins ▸ DAPPLE**.

### b. Open the wizard, load from there

```powershell
napari
```

Then **Plugins ▸ DAPPLE Wizard**. The wizard owns the loading flow and the
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

> **About `sample_type`.** It's intentionally not exposed in the form — no
> current operator consumes it. A planned spatial-filter operator (Moran's I
> with permutation null) is expected to use it to set its ON/OFF default for
> tissue vs. cell-culture data. Set it via a JSON sidecar if you need it
> already on the data model.

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

### Page 5 — Recommended workflow

The wizard reads your `ExperimentParams` and proposes a default operator chain
([details](algorithms.md#recommended-pipeline)):

1. **Reference-ion detection** — always.
2. **Empirical tolerance fit** — always.
3. **Mass recalibration** — TOF / Q-TOF families only. Per-pixel piecewise-
   linear warp from observed reference m/z values onto the consensus centroids
   (with RANSAC outlier rejection). Skipped on Orbitrap / FT-ICR where the data
   is locked enough that recalibration tends to add noise.
4. **Per-pixel normalization** — median by default. ``tic_normalize`` and
   ``reference_ion_normalize`` are also offered.
5. **Peak picking** — ``snr_peak_pick`` for centroided data, ``cwt_peak_pick``
   for profile data.
6. **Consensus peak alignment** — always.
7. **Spatial filter (Moran's I + permutation FDR)** — tissue samples only
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
- **Prevalence FDR filter** — *after the recommended chain.* Drops consensus
  channels whose prevalence is indistinguishable from random peak placement
  (permutation null on the occupancy problem, BH-FDR cutoff). An empirical
  replacement for the fixed-floor ``min_prevalence`` on the consensus card —
  better when channel peak counts span a wide range.
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

The runner caches each operator's output by `(input_hash, params_hash, op_name,
library_versions)`, so a parameter change downstream causes only the affected
nodes to re-execute on the next run. Tweaking only the consensus alignment
won't redo reference detection or peak picking.

---

## 4. The widgets

All widgets share state through a single `MsiSession`, so changes propagate
without any extra plumbing.

### Mass Spectrum Panel

A fast pyqtgraph plot of the spectrum at the cursor's pixel **or** aggregated
across foreground ROIs.

- **Raw / Harmonized** checkboxes pick the data source. Raw is the dataset's
  native peak list; Harmonized appears once consensus alignment has run.
- **Aggregation** dropdown (mean / median / sum / max) only matters in
  polygon-aggregate mode.
- **Mode** dropdown switches between *single pixel* (cursor-driven) and
  *polygon aggregate*.
- **Log y** toggles a log-scale y-axis. Stems are filtered to non-zero
  intensities first so log(0) doesn't collapse the curve.
- **Click a stem** to surface the matching consensus channel's image — see
  *Click-to-show m/z* below.

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

Once you've identified a threshold you like, switch to the wizard's Workflow
page, edit the matching operator's parameter to the value you found, and click
Run pipeline. The runner short-circuits every node whose parameters didn't
change, so re-running with one tweak is fast.

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
   and the alignment knobs (bandwidth, min cohort prevalence, recalibration,
   normalization).
3. **Discover datasets** — pre-flight: lists every matching file plus its
   pixel count without running. Use this to confirm the cohort before
   committing to a long run.
4. **Run cohort alignment** — runs on a worker thread; the log streams
   progress and the diagnostic block at the bottom shows the same
   health-rubric output you see at the end of a wizard run, scoped to the
   cohort (`shared consensus channels`, `cohort prevalence median`, etc.).

Outputs land in the chosen output directory (defaults to
`<cohort_root>/dapple_cohort/`):

- `<stem>_cohort.imzML + .ibd` per dataset
- `<stem>_cohort.tif + _channels.csv` per dataset
- `cohort_summary.json` — shared `mz_axis`, per-dataset prevalence vectors,
  cohort prevalence vector, run diagnostics, and the params used

Each per-dataset `.imzML` is a DAPPLE-harmonized export — open it later in
napari and it loads back into the Channels Panel and Spectrum Panel as a
PeakMatrix-backed dataset (consensus rows, "harmonized" toggle, etc.) without
having to re-run any pipeline.

---

## 5. Common workflows

### Drop-and-go (no wizard)

1. `napari path\to\dataset.imzML` (or directory).
2. Open **Plugins ▸ DAPPLE ▸ Channels** — you'll see the summary projections.
3. Open **Plugins ▸ DAPPLE ▸ Mass Spectrum Panel** — hover the canvas to see
   per-pixel spectra.
4. Click **Plugins ▸ DAPPLE ▸ DAPPLE Wizard** any time you want to run the
   pipeline; the wizard picks up the already-loaded dataset.

### Re-applying a saved pipeline

If a colleague hands you a `dataset.spec.xml`, drag it onto napari. The plugin
prints a summary to the console (pipeline length, experiment params, library
versions) and adds a placeholder layer with the spec attached as metadata. Open
the wizard, **Load** the original dataset, and advance to the **Workflow** page.
Cross-session restore of saved cards from the spec is planned; until then,
re-enter the parameter values from the spec into the workflow cards manually.

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

1. Add a `hot_pixel_filter` step before the recommended pipeline (custom
   pipeline support is in `pipeline/recommend.py`; surfacing this directly in
   the wizard's WorkflowPage is a future UI enhancement).
2. Pick a correction strategy: `neighbors_median` (preserves peak count by
   scaling the spike to its neighbors' median TIC) is the conservative default.
3. The diagnostic tells you how many pixels were flagged at your chosen `k_mad`
   threshold.

### Background subtraction

Once you've run consensus alignment **and** drawn at least one foreground (and
optionally one background) ROI:

1. Add a `background_subtract` step after consensus.
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

The simplest invocation is the headless CLI:

```powershell
dapple-cohort-align "C:\path\to\cohort_root" -o "C:\path\to\cohort_aligned"
```

Outputs in `cohort_aligned/`:

- `<stem>_cohort.imzML + .ibd` for each input — same m/z axis, ready to load.
- `<stem>_cohort.tif + _cohort_channels.csv` for each input — multipage TIFF
  on the shared axis, sidecar CSV with per-channel `mz` / `prevalence`.
- `cohort_summary.json` — the shared `mz_axis`, per-dataset prevalence,
  cohort prevalence, and run diagnostics.

Common flags:

```powershell
# Tighter peak picking (cohort-wide default is conservative)
dapple-cohort-align cohort_root --bandwidth-ppm 25 --min-prevalence 0.7 -o out

# Skip per-dataset MSIWarp recalibration (already-recalibrated data)
dapple-cohort-align cohort_root --no-recalibrate -o out

# Walk subdirectories
dapple-cohort-align cohort_root --recursive -o out

# Match a non-default extension
dapple-cohort-align cohort_root --pattern '*.imzml' -o out
```

The harmonization computes one shared consensus axis from all datasets pooled
together, so every output has the same number of channels in the same order.
You can stack the per-dataset `.tif` files into a `(n_datasets, H, W,
n_channels)` tensor without further alignment.

> The cohort flow is also exposed in code: ``from dapple.cohort.align import
> align_cohort, load_cohort_directory``. Use this when you want to inject the
> result into a downstream Python analysis without going through the JSON
> manifest.

### Headless reapplication via `dapple-apply-spec`

To re-run a saved pipeline on a new dataset without launching napari:

```powershell
# Apply spec to fresh dataset, write all three outputs
dapple-apply-spec "C:\path\to\new_dataset.imzML" "C:\path\to\saved.spec.xml"

# Choose where outputs go
dapple-apply-spec dataset.imzML saved.spec.xml -o out\dataset_harmonized

# TIFF only (skip imzML write)
dapple-apply-spec dataset.imzML saved.spec.xml --no-imzml -o out\

# Override the spec's RNG seed (for reproducibility experiments)
dapple-apply-spec dataset.imzML saved.spec.xml --rng-seed 42 -o out\
```

The CLI:

1. Loads the spec; reconstructs the `Pipeline` (operator chain + params + RNG
   seed) from XML.
2. Loads the input dataset (auto-detects imzML, single-CDF, or multi-CDF
   directory).
3. Runs the pipeline through the same `PipelineRunner` the wizard uses, with
   the same content-addressed cache.
4. Writes the post-pipeline dataset out as imzML, multipage TIFF, and a fresh
   `.spec.xml` capturing the rerun.

This is the right tool for batch processing (a directory of new acquisitions
all run through the same SOP), CI-style reproducibility checks (saved spec on
git-pinned data should hash bit-equal), and headless server use cases.

---

## 6. Outputs

Three formats from the **Run** page after a successful run:

### `.spec.xml`

A small XML file capturing every operator + parameter + RNG seed + library
versions, plus a content-addressed hash of the input dataset. Combine
`.spec.xml + input dataset` and DAPPLE will reproduce the run exactly. Heavy
diagnostics (bootstrap CIs, KDE curves) are referenced via a sibling Zarr if
present, so the XML stays human-readable.

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

---

## 7. Troubleshooting

### "DAPPLE Wizard" not in the Plugins menu

```powershell
python -m napari --info | grep -i dapple
```

If `dapple` isn't listed, re-install editable: `pip install -e .` from the
repo root.

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
