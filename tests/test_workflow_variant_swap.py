"""Tests for the WorkflowPage's variant-swap dropdown.

Several pipeline slots have multiple operator implementations:

- Recalibration: ``msiwarp_recalibrate`` ↔ ``lock_mass_recalibrate``
- Normalization: ``median_normalize`` ↔ ``tic_normalize`` ↔ ``reference_ion_normalize``
- Consensus: ``kde_consensus_alignment`` ↔ ``dbscan_consensus``

Each card with registered alternatives gets a Variant: dropdown that lets
the user swap the operator in-place. Selecting a different variant fires
``variant_changed(node_id, new_op_name)`` which the WorkflowPage handles by
rebuilding the pipeline with the new operator's default params for the
swapped slot, preserving every other slot's user edits.
"""

from __future__ import annotations

import pytest

pytest.importorskip("napari", reason="napari is required for widget tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


def _open_workflow_page(viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import WizardWidget, WorkflowPage

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    wiz.session.set_dataset(read_imzml(synth_centroided))
    page: WorkflowPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), WorkflowPage)
    )
    page.initializePage()
    return wiz, page


def test_recalibration_card_has_variant_dropdown(qtbot, viewer, synth_centroided):
    """The recalibration card on the synth fixture (tof_reflectron) should
    expose the MSIWarp ↔ lock-mass variant dropdown."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    recal_card = next(
        c for c in page._cards if c._node.op_name == "msiwarp_recalibrate"  # noqa: SLF001
    )
    assert recal_card._variant_combo is not None, (  # noqa: SLF001
        "msiwarp_recalibrate card must expose a variant dropdown to swap to "
        "lock_mass_recalibrate"
    )
    items = [
        recal_card._variant_combo.itemData(i)  # noqa: SLF001
        for i in range(recal_card._variant_combo.count())  # noqa: SLF001
    ]
    assert "msiwarp_recalibrate" in items
    assert "lock_mass_recalibrate" in items


def test_normalization_card_has_three_variants(qtbot, viewer, synth_centroided):
    """The normalization card should let the user pick among median / TIC /
    reference-ion."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    norm_card = next(
        c for c in page._cards if c._node.op_name == "median_normalize"  # noqa: SLF001
    )
    assert norm_card._variant_combo is not None  # noqa: SLF001
    items = [
        norm_card._variant_combo.itemData(i)  # noqa: SLF001
        for i in range(norm_card._variant_combo.count())  # noqa: SLF001
    ]
    assert set(items) == {"median_normalize", "tic_normalize", "reference_ion_normalize"}


def test_consensus_card_has_kde_dbscan_variants(qtbot, viewer, synth_centroided):
    """The consensus card should let the user pick KDE or DBSCAN."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    cons_card = next(
        c for c in page._cards if c._node.op_name == "kde_consensus_alignment"  # noqa: SLF001
    )
    assert cons_card._variant_combo is not None  # noqa: SLF001
    items = [
        cons_card._variant_combo.itemData(i)  # noqa: SLF001
        for i in range(cons_card._variant_combo.count())  # noqa: SLF001
    ]
    assert set(items) == {"kde_consensus_alignment", "dbscan_consensus"}


def test_swapping_recalibration_changes_proposed_pipeline_op_name(
    qtbot, viewer, synth_centroided
):
    """Choosing 'lock_mass_recalibrate' on the recalibration card must update
    the proposed pipeline so the recalibration node uses the new op."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    recal_card = next(
        c for c in page._cards if c._node.op_name == "msiwarp_recalibrate"  # noqa: SLF001
    )
    # Find the index of the lock_mass alternative and select it.
    target_idx = next(
        i for i in range(recal_card._variant_combo.count())  # noqa: SLF001
        if recal_card._variant_combo.itemData(i) == "lock_mass_recalibrate"  # noqa: SLF001
    )
    recal_card._variant_combo.setCurrentIndex(target_idx)  # noqa: SLF001

    # After the swap, the WorkflowPage should have rebuilt its proposed pipeline.
    op_names = [n.op_name for n in wiz._proposed_pipeline.nodes]  # noqa: SLF001
    assert "lock_mass_recalibrate" in op_names
    assert "msiwarp_recalibrate" not in op_names


def test_swapping_consensus_uses_new_default_params(qtbot, viewer, synth_centroided):
    """After swapping KDE → DBSCAN, the proposed pipeline's consensus node
    must use ``DbscanConsensusParams``, not the old ``KdeConsensusParams``."""
    from dapple.ops.dbscan_consensus import DbscanConsensusParams

    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    cons_card = next(
        c for c in page._cards if c._node.op_name == "kde_consensus_alignment"  # noqa: SLF001
    )
    target_idx = next(
        i for i in range(cons_card._variant_combo.count())  # noqa: SLF001
        if cons_card._variant_combo.itemData(i) == "dbscan_consensus"  # noqa: SLF001
    )
    cons_card._variant_combo.setCurrentIndex(target_idx)  # noqa: SLF001

    cons_node = next(
        n for n in wiz._proposed_pipeline.nodes  # noqa: SLF001
        if n.op_name == "dbscan_consensus"
    )
    assert isinstance(cons_node.params, DbscanConsensusParams), (
        "swapped node must use the new operator's params class, not the prior "
        "KDE params"
    )


def test_swapping_preserves_other_slots_user_edits(qtbot, viewer, synth_centroided):
    """Editing the picker's params, then swapping the consensus operator, must
    not clobber the picker's user-edited params."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    # Edit the picker's snr_mad to a custom value.
    pick_card = next(
        c for c in page._cards if c._node.op_name == "snr_peak_pick"  # noqa: SLF001
    )
    pick_card._inputs["snr_mad"].setValue(7.5)  # noqa: SLF001

    # Now swap consensus to DBSCAN.
    cons_card = next(
        c for c in page._cards if c._node.op_name == "kde_consensus_alignment"  # noqa: SLF001
    )
    target_idx = next(
        i for i in range(cons_card._variant_combo.count())  # noqa: SLF001
        if cons_card._variant_combo.itemData(i) == "dbscan_consensus"  # noqa: SLF001
    )
    cons_card._variant_combo.setCurrentIndex(target_idx)  # noqa: SLF001

    # After the swap, the picker's edit must survive.
    pick_card = next(
        c for c in page._cards if c._node.op_name == "snr_peak_pick"  # noqa: SLF001
    )
    assert pick_card._inputs["snr_mad"].value() == pytest.approx(7.5), (  # noqa: SLF001
        "user edits on other cards must persist across a variant swap"
    )


def test_non_variant_cards_have_no_dropdown(qtbot, viewer, synth_centroided):
    """Operators with no registered alternatives (reference detection, peak
    picking, spatial filter) should not render the Variant dropdown."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    ref_card = next(
        c for c in page._cards if c._node.op_name == "detect_reference_ions"  # noqa: SLF001
    )
    assert ref_card._variant_combo is None  # noqa: SLF001
    tol_card = next(
        c for c in page._cards
        if c._node.op_name == "empirical_tolerance_from_reference_ions"  # noqa: SLF001
    )
    assert tol_card._variant_combo is None  # noqa: SLF001


def test_variant_dropdown_carries_tooltip(qtbot, viewer, synth_centroided):
    """The variant dropdown should have a tooltip describing each
    alternative — so the user can read what each variant does."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    cons_card = next(
        c for c in page._cards if c._node.op_name == "kde_consensus_alignment"  # noqa: SLF001
    )
    tooltip = cons_card._variant_combo.toolTip()  # noqa: SLF001
    assert "KDE" in tooltip and "DBSCAN" in tooltip
