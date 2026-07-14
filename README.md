# DAPPLE

DAPPLE — a [napari](https://napari.org) plugin for mass-spectrometry imaging
(MSI) that pairs parametric defaults with empirical diagnostics, harmonizes
single images and cohorts, and supports reproducible ROI and developmental-axis
pattern analysis.

```
imzML / CDF
   │
   ├─ profile only: CWT centroiding
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  reference detection → empirical tolerance fit                  │
│       → mass recalibration → normalization                      │
│       → centroided-data SNR filter → consensus → spatial filter │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
shared m/z axis  ·  health-graded diagnostics  ·  reproducible spec.xml
   │
   ▼
imzML  ·  multipage TIFF + CSV  ·  spec.xml  ·  pattern-analysis tables
```

## Features

- **Loaders** for imzML (continuous + processed mode) and ANDI-MS NetCDF (single-file LC-MS and multi-file MSI imaging directories).
- **Wizard** that captures experiment parameters, lets you pick polygon ROIs, recommends a processing chain tuned to your instrument family + acquisition mode, and runs it on a worker thread with live progress.
- **Conservative operators** for the standard pipeline: empirical-tolerance fit (no constant-ppm assumption), MAD-based SNR or CWT peak picking, KDE or DBSCAN consensus alignment, MSIWarp or single-anchor lock-mass recalibration, Moran's I spatial filter with permutation-null FDR, hot-pixel correction, and ROI-driven background subtraction. An occupancy-based prevalence sensitivity filter is available experimentally, disabled by default, and is not a calibrated replacement for fixed ``min_prevalence``.
- **Diagnostics with health rubrics.** Every operator emits scalar metrics that the formatter renders in a tabular ✓ / ⚠ / ✗ block. Flagged metrics expand into per-state guidance: what the value means and which parameter to adjust to fix it. The command-line tools use the same format as the GUI log.
- **Per-channel viewer** (Channels Panel) — toggle, contrast, LUT-pick consensus channels and per-pixel projections; mosaic mode for side-by-side comparison.
- **Spectrum panel** with single-pixel and polygon-aggregate modes, raw vs. harmonized toggle, and click-to-show-channel drill-down.
- **Threshold Explorer** — drag a slider to preview "how many channels would survive at threshold X?" against the rejection-budget arrays of the most recent run, without re-running.
- **Cohort harmonization** — `dapple-cohort-align` CLI and `Plugins → DAPPLE → Cohort harmonization` widget align N datasets onto one shared m/z axis. Sample-balanced pooling prevents a large or bright dataset from dominating peak discovery; pixel- and dataset-prevalence are reported separately.
- **Pattern analysis** — compare named ROI unions channel by channel, or profile increasing, decreasing, endpoint-enriched, and localized patterns along an explicitly directed anatomical axis. Results are immutable tables with FDR-adjusted statistics and exportable CSV/JSON metadata.
- **Reproducibility manifest** (`.spec.xml`) capturing every operator, parameter, RNG seed, library version, input hash, and saved ROI polygon. `dapple-apply-spec` restores ROIs automatically only for a hash-matched input; registered images can opt in with `--reuse-rois`.
- **Round-trip-safe outputs** — saved imzML carries a `dapple-harmonized` marker plus axis sidecar so a saved-then-reloaded harmonized file reappears with its consensus channels intact.


## Install

DAPPLE is currently distributed as source and supports Python 3.11 and 3.12.
The `gui` extra installs the tested PyQt6 binding needed to launch napari.

### Analyst / user install

Clone the repository, then let the bootstrap helper create or reuse `.venv`,
install DAPPLE, and run the installation doctor:

```powershell
git clone https://github.com/shahlab-ucla/DAPPLE.git
cd DAPPLE
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

On macOS or Linux:

```bash
git clone https://github.com/shahlab-ucla/DAPPLE.git
cd DAPPLE
sh scripts/bootstrap.sh
```

The equivalent manual installation is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install ".[gui]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m dapple.cli.doctor
```

Use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe` on macOS or
Linux. The installed `dapple-doctor` command is an equivalent shorthand for
the final check.

### Contributor install

The contributor mode installs DAPPLE editable plus pytest, linting, typing,
benchmark, and build tools:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Dev
```

```bash
sh scripts/bootstrap.sh --dev
```

If an existing `.venv` points at a removed Python or is incomplete, the helper
stops with a repair command instead of modifying it in place. Use `-Recreate`
on PowerShell or `--recreate` on macOS/Linux to replace only the
repository-local `.venv`.

DAPPLE does not yet provide a dependency-minimal headless extra: the base
package still includes napari, QtPy, and pyqtgraph. On a CLI-only machine you
may omit `[gui]` to avoid installing a concrete PyQt6 binding, then run
`python -m dapple.cli.doctor --headless`. Here `--headless` changes validation
policy: Qt binding and widget-target failures become warnings, rather than
removing the visualization stack. The `cuda`, `directml`, and `mps`
extras are experimental backend-detection scaffolding; current DAPPLE pipeline
kernels remain CPU-bound, so installing them does not yet accelerate analysis.
imzML/CDF loading and pipeline execution currently materialize arrays in RAM;
disk-backed lazy or Zarr execution is not yet implemented.

The doctor exits nonzero if Python, a required analysis/CLI import, an installed
console launcher, the Qt binding, or napari/npe2 registration and command-target
resolution is broken. A healthy GUI installation reports three readers, seven
widgets (including cohort harmonization), and four CLI launchers.

## Getting started

### 1 — Launch napari with the wizard

Two equivalent paths:

```powershell
# A. Drop a dataset on the napari window
.\.venv\Scripts\python.exe -m napari "path\to\sample.imzML"
.\.venv\Scripts\python.exe -m napari "path\to\Boone cdf"  # multi-file CDF imaging directory

# B. Open napari, then go to Plugins → DAPPLE → DAPPLE Wizard
.\.venv\Scripts\python.exe -m napari
```

Use `.venv/bin/python -m napari` on macOS or Linux. The DAPPLE reader fires
automatically and adds a summary projection; open the wizard under
**Plugins → DAPPLE → DAPPLE Wizard**. A `.spec.xml` opened by itself shows its
metadata but does not apply the pipeline; use `dapple-apply-spec` as shown below
for a real replay.

### 2 — Walk the wizard

Seven pages, in order:

1. **Load** — pick a file or directory.
2. **Experiment parameters** — auto-populated from imzML CV terms; review the **amber** badges (defaulted) before proceeding.
3. **Preview** — click any of the six per-pixel projections to render it as a napari layer.
4. **ROI** — draw polygons on the `MSI ROIs` shapes layer; mark some as background if needed.
5. **Workflow** — review the recommended pipeline. Hover any field for a tooltip explaining what it does and how to adjust it. Three **optional cleanup operators** appear around the chain (hot-pixel correction, prevalence-FDR filter, background subtract); tick the *Enable* checkbox on each to wire it into the chain. Operators with multiple implementations (recalibration, normalization, consensus) carry a **Variant:** dropdown in their title row — swap MSIWarp ↔ lock-mass, median ↔ TIC ↔ reference-ion, or KDE ↔ DBSCAN without leaving the wizard.
6. **Review** — final summary.
7. **Run** — click **Run pipeline**. When done you see live diagnostic plots, a tabular health-rubric block in the log, a *Tune & rerun* button, and three save buttons: `.spec.xml`, `.imzML`, `.tif + _channels.csv`.

### 3 — Browse the result

After the wizard runs, four widgets become useful:

- **Channels Panel** (`Plugins → DAPPLE → Channels`) — the consensus channel rows now exist. Tick any row to add the corresponding m/z layer to the canvas; pick a saturated single-hue LUT (red/green/blue/yellow/cyan/magenta) for additive overlay; use *Show top-N by prevalence* to surface the dominant channels in one click.
- **Mass Spectrum Panel** (`Plugins → DAPPLE → Mass Spectrum Panel`) — pyqtgraph plot of the spectrum at the cursor's pixel or aggregated across foreground ROIs. Click any peak to surface the matching consensus channel in the Channels Panel.
- **Threshold Explorer** (`Plugins → DAPPLE → Threshold Explorer`) — drag the red line to preview how many channels would survive at a different prominence or prevalence threshold, without re-running.
- **Spatial and developmental patterns** (`Plugins → DAPPLE → Spatial and developmental patterns`) — run a named ROI contrast or a directed-axis profile on the harmonized channels, inspect ranked tables/plots, send a selected ion to Channels, and export CSV/JSON results.

### 4 — Harmonize a cohort

For a directory containing multiple imzML files:

```powershell
# CLI
.\.venv\Scripts\python.exe -m dapple.cli.cohort_align "path\to\cohort_root" -o "path\to\cohort_aligned"

# GUI (interactive)
.\.venv\Scripts\python.exe -m napari  # then Plugins → DAPPLE → Cohort harmonization
```

The default pool gives every dataset equal total KDE weight. Use
`--pool-weighting intensity` only when raw pooled intensity should determine
the shared peaks. `--prevalence-basis pixel` applies the prevalence threshold
to all cohort pixels; `dataset` applies it to the fraction of datasets carrying
the channel. The summary records both vectors regardless of which one filters.

Outputs in the chosen directory: per-dataset
`<stem>_cohort.imzML/.ibd/.dapple-axis.json/.tif`, plus
`cohort_summary.json` carrying an exact artifact list, `shared_consensus_mz`,
`cohort_prevalence`
(pixel-weighted), `dataset_prevalence` (one vote per dataset), per-dataset
prevalence vectors, parameters, input identities, and diagnostics.

### 5 — Analyze ROI enrichment or a developmental axis

Pattern analysis consumes a harmonized `PeakMatrix`, so run consensus alignment
or load a DAPPLE harmonized imzML first. Saved wizard ROIs can be recovered from
the accompanying `.spec.xml`:

```python
from dapple.analysis import (
    DirectedAxis,
    analyze_axis_profiles,
    analyze_roi_enrichment,
    export_analysis_result,
    rasterize_rois,
)
from dapple.io.imzml_reader import read_imzml
from dapple.io.spec_xml import read_spec_xml

ds = read_imzml("out/dataset_harmonized.imzML")
_, _, provenance = read_spec_xml("out/dataset_harmonized.spec.xml")
masks = rasterize_rois(
    ds, provenance.roi_definitions, overlap_policy="exclude"
)

# Positive log2 fold change means enriched in numerator (head) vs denominator.
roi = analyze_roi_enrichment(ds, masks, "head", "trunk")
export_analysis_result(roi, "out/patterns", stem="head_vs_trunk")

# Coordinates are zero-based napari (y, x); t=0 is anterior and t=1 posterior.
axis = DirectedAxis(
    "AP",
    start_yx=(40, 15),
    end_yx=(40, 180),
    start_label="anterior",
    end_label="posterior",
    half_width_px=25,
)
profile = analyze_axis_profiles(ds, axis, n_bins=20, rng_seed=0)
export_analysis_result(profile, "out/patterns", stem="ap_axis")
```

ROI overlap must be resolved explicitly (`error`, `exclude`, `first`, or
`allow`); contrast arms themselves may not share pixels. Axis endpoint order is
part of the hypothesis: reversing it negates trend and endpoint effects while
leaving two-sided permutation p/q values unchanged.

ROI Welch tests use pixels as units. They are useful for screening patterns
within an image, but spatial autocorrelation makes them optimistic and they are
not biological-replicate inference. For population claims, aggregate each ROI
within each independent sample and test those sample-level summaries. The core
API above is also wrapped by `dapple-analyze-patterns`:

```powershell
.\.venv\Scripts\python.exe -m dapple.cli.analyze_patterns roi `
  out\dataset_harmonized.imzML --roi-spec out\dataset_harmonized.spec.xml `
  --numerator head --denominator trunk --overlap-policy exclude -o out\patterns

.\.venv\Scripts\python.exe -m dapple.cli.analyze_patterns axis `
  out\dataset_harmonized.imzML --start 40 15 --end 40 180 `
  --start-label anterior --end-label posterior --half-width 25 -o out\patterns
```

Keep the harmonized `.imzML`, `.ibd`, and `.dapple-axis.json` together; the
`.spec.xml` is the separate source of saved ROI polygons.

### 6 — Replay a saved pipeline

```powershell
.\.venv\Scripts\python.exe -m dapple.cli.apply_spec "new_dataset.imzML" "saved.spec.xml" -o "out\dataset_harmonized"
```

Loads the spec, runs its pipeline against the new dataset, writes a fresh `.imzML + .tif + .spec.xml` plus the same diagnostic block the GUI prints.

## Documentation

- **[User guide](docs/userguide.md)** — wizard walkthrough, widget reference, common workflows, troubleshooting.
- **[Algorithm reference](docs/algorithms.md)** — what each operator computes, parameter semantics, the rationale behind every default, the diagnostic rubric, and the file-format details.


## License

MIT — see [LICENSE](LICENSE).
