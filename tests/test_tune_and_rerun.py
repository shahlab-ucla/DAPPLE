"""Regression tests for the Tune-&-Rerun flow and per-card rejection guidance.

The wizard used to feed whatever was in ``session.dataset`` back through the
operator chain on a re-run. After a successful run that's a post-consensus
``PeakMatrix``, which makes ``detect_reference_ions`` raise immediately. This
test pins down the new behaviour: re-runs always feed the original ``PeakList``
input regardless of which page the user navigates from.
"""

from __future__ import annotations

from dataclasses import replace as drep

import numpy as np
import pytest

pytest.importorskip("napari", reason="napari required for these tests")
pytest.importorskip("pytestqt", reason="pytest-qt required for these tests")


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


def _seeded_pipeline(ds_metadata):
    """Recommend pipeline tuned so consensus actually finds the synthetic peaks."""
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node

    p = recommend_pipeline(ds_metadata)
    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(
            default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5
        ),
        upstream=("pick",),
    )
    return drep(p, nodes=tuple(nodes))


# ---- snapshot: never lets a post-consensus dataset become the next input -------


def test_set_input_snapshot_refuses_peakmatrix(qtbot, viewer, synth_centroided):
    """Defensive guard: ``set_input_snapshot`` silently ignores a PeakMatrix-backed
    dataset. The pipeline must always run on the original PeakList."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.pipeline import PipelineRunner
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    raw = read_imzml(synth_centroided)
    wiz.set_input_snapshot(raw)
    assert wiz._input_dataset_snapshot is raw  # noqa: SLF001
    # Running the pipeline produces a PeakMatrix-backed result; trying to
    # snapshot that must not replace the original.
    p = _seeded_pipeline(raw.metadata)
    aligned = PipelineRunner().run(p, raw).output
    wiz.set_input_snapshot(aligned)
    assert wiz._input_dataset_snapshot is raw  # noqa: SLF001 — still the PeakList


def test_paramspage_validate_does_not_clobber_snapshot_with_post_consensus(
    qtbot, viewer, synth_centroided
):
    """The user's flow that originally crashed: load → run → Back → ParamsPage →
    edit metadata → Next → Run again. ParamsPage must update the snapshot's
    metadata in place rather than capturing the post-consensus dataset that's
    currently sitting in ``session.dataset``.
    """
    from dapple.io.imzml_reader import read_imzml
    from dapple.pipeline import PipelineRunner
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import ParamsPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    raw = read_imzml(synth_centroided)
    wiz.session.set_dataset(raw)
    wiz.set_input_snapshot(raw)

    # Run the pipeline to populate session.dataset with a PeakMatrix.
    p = _seeded_pipeline(raw.metadata)
    aligned = PipelineRunner().run(p, raw).output
    wiz.session.set_dataset(aligned)

    # Now the user navigates Back and edits something on ParamsPage. validatePage
    # is what actually commits the change; calling it directly is the cleanest
    # test of the snapshot logic.
    params_page: ParamsPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), ParamsPage)
    )
    params_page.initializePage()
    assert params_page.validatePage()

    # The snapshot must STILL be a PeakList (not the post-consensus PeakMatrix
    # that's currently sitting on the session).
    from dapple.data.dataset import PeakList

    assert isinstance(wiz._input_dataset_snapshot.backend, PeakList), (  # noqa: SLF001
        f"snapshot got clobbered to {type(wiz._input_dataset_snapshot.backend).__name__}"  # noqa: SLF001
    )


def test_rerun_after_harmonization_does_not_raise(qtbot, viewer, synth_centroided):
    """End-to-end: load → run → tweak param → re-run. Must succeed (the original
    bug was an immediate ``NotImplementedError`` from detect_reference_ions when
    fed a PeakMatrix)."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline.pipeline import Node
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    raw = read_imzml(synth_centroided)
    wiz.session.set_dataset(raw)
    wiz.set_input_snapshot(raw)
    wiz._proposed_pipeline = _seeded_pipeline(raw.metadata)  # noqa: SLF001

    run_page: RunPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), RunPage)
    )
    # First run.
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001
    assert run_page._run_result is not None  # noqa: SLF001

    # Tweak the consensus parameter and re-run. The previous bug surfaced here
    # because session.dataset is now a PeakMatrix and (without the snapshot fix)
    # the runner would feed it back through detect_reference_ions.
    new_consensus = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(
            default_tol_ppm=200.0, bandwidth_ppm=15.0, min_prevalence=0.5
        ),
        upstream=("pick",),
    )
    wiz._proposed_pipeline = drep(  # noqa: SLF001
        wiz._proposed_pipeline,  # noqa: SLF001
        nodes=tuple([*wiz._proposed_pipeline.nodes[:-1], new_consensus]),  # noqa: SLF001
    )
    # initializePage flips the save buttons off (parameters changed).
    run_page.initializePage()
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001
    # No exception → re-run worked.
    assert run_page._run_result is not None  # noqa: SLF001


# ---- Tune & Rerun navigation ---------------------------------------------------


def test_tune_button_disabled_until_run_completes(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    wiz.session.set_dataset(read_imzml(synth_centroided))
    run_page: RunPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), RunPage)
    )
    assert not run_page._tune_btn.isEnabled()  # noqa: SLF001


def test_tune_button_navigates_back_to_workflow(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget, WorkflowPage

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    raw = read_imzml(synth_centroided)
    wiz.session.set_dataset(raw)
    wiz.set_input_snapshot(raw)
    wiz._proposed_pipeline = _seeded_pipeline(raw.metadata)  # noqa: SLF001

    # The wizard's navigation stack is only meaningful once it's shown. show()
    # initializes the start page; we then walk forward via next() until we land
    # on the RunPage, simulating the user clicking through.
    wiz.show()
    qtbot.waitExposed(wiz)
    # Mark LoadPage 'complete' since we already loaded the dataset by hand.
    from dapple.widgets.wizard import LoadPage

    load_page = next(wiz.page(i) for i in range(7) if isinstance(wiz.page(i), LoadPage))
    load_page._loaded = True  # noqa: SLF001
    load_page.completeChanged.emit()
    # Walk forward via QWizard.next(); it returns None, so we use currentId() to
    # detect when we've stopped advancing.
    last_id = -1
    for _ in range(20):
        if isinstance(wiz.currentPage(), RunPage):
            break
        cur = wiz.currentId()
        if cur == last_id:
            break
        last_id = cur
        wiz.next()

    run_page = next(wiz.page(i) for i in range(7) if isinstance(wiz.page(i), RunPage))
    assert isinstance(wiz.currentPage(), RunPage), (
        f"failed to navigate to RunPage; ended on {type(wiz.currentPage()).__name__}"
    )
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001

    run_page._on_tune_and_rerun()  # noqa: SLF001
    assert isinstance(wiz.currentPage(), WorkflowPage)


# ---- Per-card rejection guidance -----------------------------------------------


def test_workflow_card_shows_rejection_guidance_after_run(
    qtbot, viewer, synth_centroided
):
    """After a successful run, navigating back to the WorkflowPage rebuilds the
    cards. The card for the most-filter-heavy operator (consensus) should now
    contain a guidance box that summarises rejection counts and points at the
    parameter most worth tuning."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.pipeline import PipelineRunner
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import WizardWidget, WorkflowPage

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    raw = read_imzml(synth_centroided)
    wiz.session.set_dataset(raw)
    wiz.set_input_snapshot(raw)
    wiz._proposed_pipeline = _seeded_pipeline(raw.metadata)  # noqa: SLF001
    # Run once to populate diagnostics on the wizard.
    result = PipelineRunner().run(wiz._proposed_pipeline, raw)  # noqa: SLF001
    wiz._last_run_result = result  # noqa: SLF001

    # Now visit the workflow page; its initializePage should pull diagnostics off
    # the wizard and decorate each card.
    wf_page: WorkflowPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), WorkflowPage)
    )
    wf_page.initializePage()
    consensus_card = next(c for c in wf_page._cards if c._node.id == "consensus")  # noqa: SLF001
    # Locate the guidance label among the card's children.
    from qtpy.QtWidgets import QLabel

    guidance_labels = [
        w
        for w in consensus_card.findChildren(QLabel)
        if w.text() and "consensus peaks" in w.text().lower()
    ]
    assert guidance_labels, "no rejection-budget guidance rendered on consensus card"
    text = guidance_labels[0].text()
    # On the synthetic fixture every planted peak survives all filters cleanly,
    # so the rejection contributors don't surface; the guidance still has to at
    # least record the headline counts. The dominant-contributor selection logic
    # is exercised by test_summarize_rejection_budget_consensus_picks_dominant_contributor.
    assert "consensus peaks from" in text.lower()


def test_summarize_rejection_budget_handles_unknown_operator():
    """Unknown operators emit no guidance (rather than blowing up)."""
    from dapple.widgets.wizard import _summarize_rejection_budget

    out = _summarize_rejection_budget("does_not_exist", {"foo": 1.0})
    assert out == []


def test_summarize_rejection_budget_consensus_picks_dominant_contributor():
    """When prominence rejects more than prevalence, guidance points at prominence."""
    from dapple.widgets.wizard import _summarize_rejection_budget

    summary = {
        "n_consensus_peaks": 50,
        "n_local_maxima_total": 1000,
        "n_post_prominence": 100,
        "n_rejected_by_prominence": 900,
        "n_rejected_by_prevalence": 50,
    }
    out = _summarize_rejection_budget("kde_consensus_alignment", summary)
    text = " ".join(out).lower()
    assert "prominence threshold rejected" in text
    assert "min_prominence_quantile" in text


def test_summarize_rejection_budget_consensus_flips_when_prevalence_dominates():
    """When prevalence rejects more than prominence, guidance flips."""
    from dapple.widgets.wizard import _summarize_rejection_budget

    summary = {
        "n_consensus_peaks": 5,
        "n_local_maxima_total": 100,
        "n_post_prominence": 80,
        "n_rejected_by_prominence": 20,
        "n_rejected_by_prevalence": 75,
    }
    out = _summarize_rejection_budget("kde_consensus_alignment", summary)
    text = " ".join(out).lower()
    assert "prevalence filter rejected" in text
    assert "min_prevalence" in text
