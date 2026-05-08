# DAPPLE

DAPPLE — a [napari](https://napari.org) plugin for mass-spectrometry imaging (MSI) that pairs every parametric default with an empirical diagnostic, runs reproducible processing chains, and harmonizes data across single images and cohorts.

```
imzML / CDF
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  reference detection → empirical tolerance fit                  │
│       → mass recalibration → normalization → peak picking       │
│       → consensus alignment → spatial filter                    │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
shared m/z axis  ·  health-graded diagnostics  ·  reproducible spec.xml
   │
   ▼
imzML  ·  multipage TIFF + CSV  ·  spec.xml
```

## Features

- **Loaders** for imzML (continuous + processed mode) and ANDI-MS NetCDF (single-file LC-MS and multi-file MSI imaging directories).
- **Wizard** that captures experiment parameters, lets you pick polygon ROIs, recommends a processing chain tuned to your instrument family + acquisition mode, and runs it on a worker thread with live progress.
- **Conservative operators** for the standard pipeline: empirical-tolerance fit (no constant-ppm assumption), MAD-based SNR or CWT peak picking, KDE or DBSCAN consensus alignment, MSIWarp or single-anchor lock-mass recalibration, Moran's I spatial filter with permutation-null FDR, hot-pixel correction, and ROI-driven background subtraction.
- **Diagnostics with health rubrics.** Every operator emits scalar metrics that the formatter renders in a tabular ✓ / ⚠ / ✗ block. Flagged metrics expand into per-state guidance: what the value means and which parameter to adjust to fix it. Same format in the GUI log and the headless CLIs.
- **Per-channel viewer** (Channels Panel) — toggle, contrast, LUT-pick consensus channels and per-pixel projections; mosaic mode for side-by-side comparison.
- **Spectrum panel** with single-pixel and polygon-aggregate modes, raw vs. harmonized toggle, and click-to-show-channel drill-down.
- **Threshold Explorer** — drag a slider to preview "how many channels would survive at threshold X?" against the rejection-budget arrays of the most recent run, without re-running.
- **Cohort harmonization** — `dapple-cohort-align` CLI and `Plugins → DAPPLE → Cohort harmonization` widget align N datasets onto one shared m/z axis via pooled-consensus, with per-dataset and cohort-wide prevalence diagnostics.
- **Reproducibility manifest** (`.spec.xml`) capturing every operator, parameter, RNG seed, library version, and input hash. `dapple-apply-spec` reapplies a saved spec to a fresh dataset headless.
- **Round-trip-safe outputs** — saved imzML carries a `dapple-harmonized` marker plus axis sidecar so a saved-then-reloaded harmonized file reappears with its consensus channels intact.


## Install

DAPPLE is currently distributed as source. Python 3.12+ recommended.

```powershell
git clone https://github.com/shahlab-ucla/DAPPLE.git
cd DAPPLE
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

pip install -e ".[dev]"           # editable install with test dependencies
```

The `[dev]` extra pulls in pytest + pytest-qt + ruff + mypy. Drop it for a runtime-only install.

Confirm DAPPLE is registered as a napari plugin:

```powershell
python -c "from npe2 import PluginManager; pm = PluginManager.instance(); pm.discover(); print(pm.get_manifest('dapple'))"
```

You should see `dapple (DAPPLE)` listed with three readers and several widgets.

## Getting started

### 1 — Launch napari with the wizard

Two equivalent paths:

```powershell
# A. Drop a dataset on the napari window
napari "path\to\sample.imzML"
napari "path\to\Boone cdf"           # multi-file CDF imaging directory
napari "path\to\sample.spec.xml"     # re-applies a saved pipeline

# B. Open napari, then go to Plugins → DAPPLE Wizard
napari
```

The DAPPLE reader fires automatically; a peak-count projection lands on the canvas, and the wizard becomes available under **Plugins → DAPPLE Wizard**.

### 2 — Walk the wizard

Seven pages, in order:

1. **Load** — pick a file or directory.
2. **Experiment parameters** — auto-populated from imzML CV terms; review the **amber** badges (defaulted) before proceeding.
3. **Preview** — click any of the six per-pixel projections to render it as a napari layer.
4. **ROI** — draw polygons on the `MSI ROIs` shapes layer; mark some as background if needed.
5. **Workflow** — review the recommended pipeline. Hover any field for a tooltip explaining what it does and how to adjust it. Two **optional cleanup operators** appear at the top (hot-pixel) and bottom (background subtract); tick the *Enable* checkbox to wire them into the chain.
6. **Review** — final summary.
7. **Run** — click **Run pipeline**. When done you see live diagnostic plots, a tabular health-rubric block in the log, a *Tune & rerun* button, and three save buttons: `.spec.xml`, `.imzML`, `.tif + _channels.csv`.

### 3 — Browse the result

After the wizard runs, three widgets become useful:

- **Channels Panel** (`Plugins → DAPPLE → Channels`) — the consensus channel rows now exist. Tick any row to add the corresponding m/z layer to the canvas; pick a saturated single-hue LUT (red/green/blue/yellow/cyan/magenta) for additive overlay; use *Show top-N by prevalence* to surface the dominant channels in one click.
- **Mass Spectrum Panel** (`Plugins → DAPPLE → Mass Spectrum Panel`) — pyqtgraph plot of the spectrum at the cursor's pixel or aggregated across foreground ROIs. Click any peak to surface the matching consensus channel in the Channels Panel.
- **Threshold Explorer** (`Plugins → DAPPLE → Threshold Explorer`) — drag the red line to preview how many channels would survive at a different prominence or prevalence threshold, without re-running.

### 4 — Harmonize a cohort

For a directory containing multiple imzML files:

```powershell
# CLI
dapple-cohort-align "path\to\cohort_root" -o "path\to\cohort_aligned"

# GUI (interactive)
napari       # then Plugins → DAPPLE → Cohort harmonization
```

Outputs in the chosen directory: per-dataset `<stem>_cohort.imzML/.ibd/.tif`, plus a single `cohort_summary.json` carrying the shared `mz_axis`, per-dataset prevalence vectors, and the run diagnostics.

### 5 — Replay a saved pipeline

```powershell
dapple-apply-spec "new_dataset.imzML" "saved.spec.xml" -o "out\dataset_harmonized"
```

Loads the spec, runs its pipeline against the new dataset, writes a fresh `.imzML + .tif + .spec.xml` plus the same diagnostic block the GUI prints.

## Documentation

- **[User guide](docs/userguide.md)** — wizard walkthrough, widget reference, common workflows, troubleshooting.
- **[Algorithm reference](docs/algorithms.md)** — what each operator computes, parameter semantics, the rationale behind every default, the diagnostic rubric, and the file-format details.


## License

MIT — see [LICENSE](LICENSE).
