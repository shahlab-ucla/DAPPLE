"""Headless command-line tools for DAPPLE.

These entry points let scripts and pipelines run DAPPLE without a napari GUI:

- ``dapple-apply-spec`` — apply a saved ``.spec.xml`` to an input dataset and
  write the harmonized outputs.
- ``dapple-cohort-align`` — run a unified consensus alignment across a directory
  of datasets and emit one shared TIFF + per-dataset diagnostics.

Both commands print a brief progress summary to stdout. Exit code is 0 on
success, 1 on a load / pipeline error, 2 on bad arguments.
"""
