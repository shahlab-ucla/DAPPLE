from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pytestqt", reason="pytest-qt required for widget tests")


def _aligned_with_rois():
    from dapple.data.dataset import MSIDataset, PeakMatrix
    from dapple.data.metadata import DatasetIdentity, ExperimentParams, RoiDef

    h = w = 4
    coords = np.asarray(
        [(x + 1, y + 1) for y in range(h) for x in range(w)], dtype=np.int32
    )
    matrix = np.zeros((h * w, 3), dtype=np.float32)
    matrix[:, 2] = 5.0
    matrix[coords[:, 0] <= 2, 0] = 20.0
    matrix[coords[:, 0] >= 3, 1] = 25.0
    rois = (
        RoiDef(
            name="early",
            vertices=((0.0, 0.0), (0.0, 1.4), (3.0, 1.4), (3.0, 0.0)),
        ),
        RoiDef(
            name="late",
            vertices=((0.0, 1.6), (0.0, 3.0), (3.0, 3.0), (3.0, 1.6)),
        ),
    )
    return MSIDataset(
        coords=coords,
        grid_shape=(h, w),
        metadata=ExperimentParams(
            instrument_family="qtof",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="positive",
            mz_min=100.0,
            mz_max=400.0,
            sample_type="tissue",
        ),
        backend=PeakMatrix(
            matrix=matrix,
            mz_axis=np.asarray([150.0, 250.0, 350.0]),
        ),
        identity=DatasetIdentity(
            source_path="development.imzML",
            content_sha256=hashlib.sha256(b"development").hexdigest(),
        ),
        rois=rois,
    )


def test_widget_runs_roi_contrast_synchronously(qtbot):
    from dapple.analysis import RoiEnrichmentResult
    from dapple.widgets._session import MsiSession
    from dapple.widgets.developmental import DevelopmentalAnalysisWidget

    session = MsiSession()
    session.set_dataset(_aligned_with_rois())
    widget = DevelopmentalAnalysisWidget(session=session)
    qtbot.addWidget(widget)
    result = widget._compute()  # noqa: SLF001
    assert isinstance(result, RoiEnrichmentResult)
    assert result.log2_fold_change[0] > 0
    assert result.log2_fold_change[1] < 0


def test_widget_runs_directed_axis_synchronously(qtbot):
    from dapple.analysis import AxisProfileResult
    from dapple.widgets._session import MsiSession
    from dapple.widgets.developmental import DevelopmentalAnalysisWidget

    session = MsiSession()
    session.set_dataset(_aligned_with_rois())
    widget = DevelopmentalAnalysisWidget(session=session)
    qtbot.addWidget(widget)
    widget._tabs.setCurrentIndex(1)  # noqa: SLF001
    widget._start_y.setValue(1.5)  # noqa: SLF001
    widget._start_x.setValue(0.0)  # noqa: SLF001
    widget._end_y.setValue(1.5)  # noqa: SLF001
    widget._end_x.setValue(3.0)  # noqa: SLF001
    widget._n_bins.setValue(4)  # noqa: SLF001
    widget._permutations.setValue(9)  # noqa: SLF001
    result = widget._compute()  # noqa: SLF001
    assert isinstance(result, AxisProfileResult)
    assert result.spearman_rho[0] < 0
    assert result.spearman_rho[1] > 0


def test_developmental_widget_is_contributed_to_napari_manifest():
    manifest = Path("src/dapple/napari.yaml").read_text(encoding="utf-8")
    assert "dapple.open_developmental" in manifest
    assert "dapple.widgets.developmental:DevelopmentalAnalysisWidget" in manifest


def test_widget_invalidates_stale_result_after_dataset_change(qtbot):
    from dataclasses import replace

    from dapple.widgets._session import MsiSession
    from dapple.widgets.developmental import DevelopmentalAnalysisWidget

    session = MsiSession()
    dataset = _aligned_with_rois()
    session.set_dataset(dataset)
    widget = DevelopmentalAnalysisWidget(session=session)
    qtbot.addWidget(widget)
    result = widget._compute()  # noqa: SLF001
    widget._analysis_generation += 1  # noqa: SLF001
    token = widget._analysis_generation  # noqa: SLF001
    widget._accept_result(token, result)  # noqa: SLF001
    assert widget._result is result  # noqa: SLF001
    assert widget._export_btn.isEnabled()  # noqa: SLF001

    session.set_dataset(replace(dataset, rng_seed=1))
    assert widget._result is None  # noqa: SLF001
    assert not widget._export_btn.isEnabled()  # noqa: SLF001

    # A late return from the old worker must not restore the stale result.
    widget._accept_result(token, result)  # noqa: SLF001
    assert widget._result is None  # noqa: SLF001
