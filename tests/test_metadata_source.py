"""Tests for per-field metadata source tracking and sidecar discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dapple.io.cdf_image_reader import read_cdf_image
from dapple.io.cdf_reader import read_cdf
from dapple.io.imzml_reader import read_imzml


# ---- imzML reader tracks per-field sources --------------------------------------


def test_imzml_tracks_per_field_metadata_source(synth_centroided):
    ds = read_imzml(synth_centroided)
    src = ds.extra["metadata_source"]
    # Synth fixture declares MS:1000084, MS:1000075, MS:1000127, MS:1000129, pixel size,
    # and lacks per-spectrum lowest/highest m/z — so polarity/mode/ionization come from
    # imzML CV terms but mz_min/mz_max fall back to the default.
    assert src["instrument_family"] == "imzml"
    assert src["ionization"] == "imzml"
    assert src["profile_or_centroided"] == "imzml"
    assert src["polarity"] == "imzml"
    assert src["pixel_size_um"] == "imzml"
    assert src["mz_min"] == "default"
    assert src["mz_max"] == "default"
    # sample_type and notes are never auto-detected.
    assert src["sample_type"] == "default"
    assert src["notes"] == "default"


def test_imzml_no_sidecar_path_when_none_present(synth_centroided):
    ds = read_imzml(synth_centroided)
    assert ds.extra.get("sidecar_path") is None


# ---- JSON sidecar overlay -------------------------------------------------------


def test_imzml_json_sidecar_overrides_detected_fields(synth_centroided, tmp_path):
    """A `<stem>.metadata.json` should override matching fields and tag them 'sidecar'."""
    base = synth_centroided.with_suffix("")
    sidecar = base.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "instrument_family": "qtof",  # overrides the auto-detected tof_axial
                "sample_type": "tissue",
                "mz_min": 50.0,
                "mz_max": 800.0,
                "notes": "added by sidecar",
            }
        ),
        encoding="utf-8",
    )
    try:
        ds = read_imzml(synth_centroided)
        assert ds.metadata.instrument_family == "qtof"
        assert ds.metadata.sample_type == "tissue"
        assert ds.metadata.mz_min == 50.0
        assert ds.metadata.mz_max == 800.0
        assert ds.metadata.notes == "added by sidecar"
        src = ds.extra["metadata_source"]
        assert src["instrument_family"] == "sidecar"
        assert src["sample_type"] == "sidecar"
        assert src["mz_min"] == "sidecar"
        assert src["mz_max"] == "sidecar"
        # Polarity wasn't in the sidecar — keep the auto-detected source.
        assert src["polarity"] == "imzml"
        # Path is recorded.
        assert ds.extra["sidecar_path"] == str(sidecar)
    finally:
        if sidecar.exists():
            sidecar.unlink()


def test_imzml_json_sidecar_ignores_unknown_keys(synth_centroided):
    base = synth_centroided.with_suffix("")
    sidecar = base.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps({"not_a_field": 42, "ionization": "desi"}), encoding="utf-8"
    )
    try:
        ds = read_imzml(synth_centroided)
        assert ds.metadata.ionization == "desi"
        # Unknown keys don't end up in source.
        src = ds.extra["metadata_source"]
        assert "not_a_field" not in src
    finally:
        sidecar.unlink()


def test_imzml_experiment_json_alias(synth_centroided):
    base = synth_centroided.with_suffix("")
    sidecar = base.with_suffix(".experiment.json")
    sidecar.write_text(json.dumps({"sample_type": "cell_culture"}), encoding="utf-8")
    try:
        ds = read_imzml(synth_centroided)
        assert ds.metadata.sample_type == "cell_culture"
        assert ds.extra["metadata_source"]["sample_type"] == "sidecar"
    finally:
        sidecar.unlink()


# ---- .spec.xml sidecar overlay --------------------------------------------------


def test_imzml_spec_xml_sidecar_overrides(synth_centroided, tmp_path):
    """If a `<stem>.spec.xml` is present, its experimentParams overlay the imzML."""
    from dapple.data.metadata import ExperimentParams
    from dapple.io.spec_xml import make_provenance, write_spec_xml
    from dapple.pipeline import recommend_pipeline

    # Build a representative spec.xml next to the synth dataset.
    base = synth_centroided.with_suffix("")
    spec_path = base.with_suffix(".spec.xml")
    custom_ep = ExperimentParams(
        instrument_family="orbitrap",
        ionization="esi",
        profile_or_centroided="profile",
        polarity="positive",
        mz_min=100.0,
        mz_max=2000.0,
        pixel_size_um=25.0,
        sample_type="tissue",
    )
    pipeline = recommend_pipeline(custom_ep)
    write_spec_xml(
        spec_path,
        pipeline=pipeline,
        experiment_params=custom_ep,
        provenance=make_provenance(plugin_version="0.1.0.dev0", input_dataset_hash="x"),
    )
    try:
        ds = read_imzml(synth_centroided)
        assert ds.metadata.instrument_family == "orbitrap"
        assert ds.metadata.sample_type == "tissue"
        src = ds.extra["metadata_source"]
        assert src["instrument_family"] == "spec_xml"
        assert src["sample_type"] == "spec_xml"
        assert ds.extra["sidecar_path"] == str(spec_path)
    finally:
        spec_path.unlink()


def test_imzml_json_sidecar_takes_priority_over_spec_xml(synth_centroided):
    """If both .metadata.json and .spec.xml exist, JSON wins (it's hand-edited)."""
    from dapple.data.metadata import ExperimentParams
    from dapple.io.spec_xml import make_provenance, write_spec_xml
    from dapple.pipeline import recommend_pipeline

    base = synth_centroided.with_suffix("")
    spec_path = base.with_suffix(".spec.xml")
    json_path = base.with_suffix(".metadata.json")

    custom_ep = ExperimentParams(
        instrument_family="orbitrap",
        ionization="esi",
        profile_or_centroided="profile",
        polarity="positive",
        mz_min=100.0,
        mz_max=2000.0,
    )
    write_spec_xml(
        spec_path,
        pipeline=recommend_pipeline(custom_ep),
        experiment_params=custom_ep,
        provenance=make_provenance(plugin_version="0.1.0.dev0", input_dataset_hash="x"),
    )
    json_path.write_text(json.dumps({"instrument_family": "qtof"}), encoding="utf-8")
    try:
        ds = read_imzml(synth_centroided)
        # JSON wins.
        assert ds.metadata.instrument_family == "qtof"
        assert ds.extra["metadata_source"]["instrument_family"] == "sidecar"
        assert ds.extra["sidecar_path"] == str(json_path)
    finally:
        spec_path.unlink()
        json_path.unlink()


# ---- CDF readers ---------------------------------------------------------------


def test_cdf_single_file_tracks_source(synth_lcms_cdf):
    with pytest.warns(UserWarning):
        ds = read_cdf(synth_lcms_cdf)
    src = ds.extra["metadata_source"]
    assert src["polarity"] == "andims"
    assert src["mz_min"] == "andims"
    assert src["mz_max"] == "andims"
    assert src["sample_type"] == "default"


def test_cdf_image_directory_sidecar(synth_cdf_image_dir):
    """metadata.json placed inside the directory should overlay the merged CDF metadata."""
    sidecar = synth_cdf_image_dir / "metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "sample_type": "tissue",
                "pixel_size_um": 75.0,
                "instrument_family": "qtof",
            }
        ),
        encoding="utf-8",
    )
    try:
        ds = read_cdf_image(synth_cdf_image_dir)
        assert ds.metadata.sample_type == "tissue"
        assert ds.metadata.pixel_size_um == 75.0
        assert ds.metadata.instrument_family == "qtof"
        src = ds.extra["metadata_source"]
        assert src["sample_type"] == "sidecar"
        assert src["pixel_size_um"] == "sidecar"
        assert src["instrument_family"] == "sidecar"
        assert ds.extra["sidecar_path"] == str(sidecar)
    finally:
        sidecar.unlink()


# ---- ParamsPage UI ------------------------------------------------------------


def test_params_page_renders_per_field_badges(qtbot, make_napari_viewer, synth_centroided):
    from dapple.widgets.wizard import ParamsPage, WizardWidget

    viewer = make_napari_viewer()
    wiz = WizardWidget(napari_viewer=viewer)
    qtbot.addWidget(wiz)
    ds = read_imzml(synth_centroided)
    wiz.session.set_dataset(ds)

    page: ParamsPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), ParamsPage)
    )
    page.initializePage()
    # Auto-detected fields should show the imzml badge text; defaulted fields the warn one.
    assert "auto-detected (imzML)" in page._badges["polarity"].text()  # noqa: SLF001
    assert "default — please verify" in page._badges["mz_min"].text()  # noqa: SLF001


def test_params_page_user_edit_flips_badge_to_override(qtbot, make_napari_viewer, synth_centroided):
    from dapple.widgets.wizard import ParamsPage, WizardWidget

    viewer = make_napari_viewer()
    wiz = WizardWidget(napari_viewer=viewer)
    qtbot.addWidget(wiz)
    ds = read_imzml(synth_centroided)
    wiz.session.set_dataset(ds)
    page: ParamsPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), ParamsPage)
    )
    page.initializePage()

    # Change polarity from negative -> positive.
    page._polarity.setCurrentText("positive")  # noqa: SLF001
    assert "user override" in page._badges["polarity"].text()  # noqa: SLF001

    # Click reset.
    page._reset_btn.click()  # noqa: SLF001
    assert page._polarity.currentText() == "negative"  # noqa: SLF001
    assert "auto-detected (imzML)" in page._badges["polarity"].text()  # noqa: SLF001


def test_params_page_validate_records_user_overrides_in_extra(
    qtbot, make_napari_viewer, synth_centroided
):
    from dapple.widgets.wizard import ParamsPage, WizardWidget

    viewer = make_napari_viewer()
    wiz = WizardWidget(napari_viewer=viewer)
    qtbot.addWidget(wiz)
    wiz.session.set_dataset(read_imzml(synth_centroided))
    page: ParamsPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), ParamsPage)
    )
    page.initializePage()
    page._polarity.setCurrentText("positive")  # noqa: SLF001
    assert page.validatePage() is True
    src = wiz.session.dataset.extra["metadata_source"]
    assert src["polarity"] == "user"
    # Untouched fields keep their original source.
    assert src["instrument_family"] == "imzml"


@pytest.mark.real_data
@pytest.mark.skipif(
    __import__("os").environ.get("MSI_REAL_DATA", "0") != "1",
    reason="real-data only",
)
def test_real_jerboa_metadata_source_includes_imzml_terms(real_dataset_dir: Path):
    p = real_dataset_dir / "jerboa-100825.imzML"
    if not p.exists():
        pytest.skip(f"{p} not found")
    ds = read_imzml(p)
    src = ds.extra["metadata_source"]
    # Real Bruker SCiLS export populates instrument_family, ionization, mode, polarity,
    # pixel_size_um, and the per-spectrum m/z bounds — every one of these should be
    # auto-detected.
    for fname in (
        "instrument_family",
        "ionization",
        "profile_or_centroided",
        "polarity",
        "mz_min",
        "mz_max",
        "pixel_size_um",
    ):
        assert src[fname] == "imzml", f"{fname!r} expected 'imzml' source, got {src[fname]!r}"


@pytest.mark.real_data
@pytest.mark.skipif(
    __import__("os").environ.get("MSI_REAL_DATA", "0") != "1",
    reason="real-data only",
)
def test_real_boone_metadata_source_includes_andims(real_dataset_dir: Path):
    boone = real_dataset_dir / "Boone cdf"
    if not boone.exists():
        pytest.skip(f"{boone} not found")
    ds = read_cdf_image(boone)
    src = ds.extra["metadata_source"]
    for fname in (
        "ionization",
        "profile_or_centroided",
        "polarity",
        "mz_min",
        "mz_max",
    ):
        assert src[fname] == "andims", f"{fname!r} expected 'andims' source, got {src[fname]!r}"
