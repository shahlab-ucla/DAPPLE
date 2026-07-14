from __future__ import annotations

from dataclasses import replace

import numpy as np


def test_session_preserves_true_raw_input_across_harmonization(synth_centroided):
    from dapple.data.dataset import PeakMatrix
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession

    raw = read_imzml(synth_centroided)
    processed = replace(
        raw,
        backend=PeakMatrix(
            matrix=np.ones((raw.n_pixels, 2), dtype=np.float32),
            mz_axis=np.array([200.0, 300.0]),
        ),
    )
    session = MsiSession()
    session.set_dataset(raw)
    session.set_dataset(processed)
    assert session.dataset is processed
    assert session.raw_dataset is raw


def test_session_clears_stale_raw_for_unrelated_processed_file(synth_centroided):
    from dapple.data.dataset import PeakMatrix
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession

    raw = read_imzml(synth_centroided)
    processed = replace(
        raw,
        identity=replace(raw.identity, source_path="different.imzML"),
        backend=PeakMatrix(
            matrix=np.ones((raw.n_pixels, 1), dtype=np.float32),
            mz_axis=np.array([250.0]),
        ),
    )
    session = MsiSession()
    session.set_dataset(raw)
    session.set_dataset(processed)
    assert session.raw_dataset is None
