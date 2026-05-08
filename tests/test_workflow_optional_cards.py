"""Tests for the WorkflowPage's optional-cleanup-operator UI.

The recommended pipeline doesn't include ``hot_pixel_filter`` or
``background_subtract`` by default — they're surfaced as opt-in cards on the
WorkflowPage with an Enable checkbox. Users tick the box to wire the operator
into the pipeline; toggling it off removes it without affecting the rest of
the chain.
"""

from __future__ import annotations

import pytest

pytest.importorskip("napari", reason="napari is required for widget tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")


@pytest.fixture
def viewer(make_napari_viewer):
    v = make_napari_viewer()
    yield v


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


def test_workflow_page_renders_optional_hot_pixel_card(qtbot, viewer, synth_centroided):
    """Hot-pixel filter card appears at the front of the cards list, optional + disabled."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    op_names = [c._node.op_name for c in page._cards]  # noqa: SLF001
    assert op_names[0] == "hot_pixel_filter"
    assert page._cards[0].is_optional()  # noqa: SLF001
    assert not page._cards[0].is_enabled()  # noqa: SLF001 — default off


def test_workflow_page_renders_optional_background_subtract_card(
    qtbot, viewer, synth_centroided
):
    """Background-subtract card appears at the END, optional + disabled."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    op_names = [c._node.op_name for c in page._cards]  # noqa: SLF001
    assert op_names[-1] == "background_subtract"
    assert page._cards[-1].is_optional()  # noqa: SLF001
    assert not page._cards[-1].is_enabled()  # noqa: SLF001


def test_disabled_optional_cards_are_excluded_from_pipeline(
    qtbot, viewer, synth_centroided
):
    """validatePage skips optional cards whose checkbox is unchecked."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)
    page.validatePage()
    op_names = [n.op_name for n in wiz._proposed_pipeline.nodes]  # noqa: SLF001
    assert "hot_pixel_filter" not in op_names
    assert "background_subtract" not in op_names


def test_enabling_hot_pixel_card_inserts_it_at_front_of_pipeline(
    qtbot, viewer, synth_centroided
):
    """Toggling the hot-pixel checkbox adds it to the pipeline ahead of detect_reference_ions."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)

    hot_card = next(c for c in page._cards if c._node.op_name == "hot_pixel_filter")  # noqa: SLF001
    hot_card._enable_checkbox.setChecked(True)  # noqa: SLF001
    page.validatePage()
    op_names = [n.op_name for n in wiz._proposed_pipeline.nodes]  # noqa: SLF001
    assert op_names[0] == "hot_pixel_filter"
    assert op_names[1] == "detect_reference_ions"


def test_enabled_hot_pixel_card_threads_upstream_correctly(
    qtbot, viewer, synth_centroided
):
    """The recommended pipeline's first node has its upstream re-pointed to hot_pixel_filter
    when the hot-pixel card is enabled."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)

    hot_card = next(c for c in page._cards if c._node.op_name == "hot_pixel_filter")  # noqa: SLF001
    hot_card._enable_checkbox.setChecked(True)  # noqa: SLF001
    page.validatePage()
    nodes = wiz._proposed_pipeline.nodes  # noqa: SLF001
    # First node has empty upstream.
    assert nodes[0].upstream == ()
    # Second node (detect_reference_ions) now points at the hot-pixel node.
    assert nodes[1].upstream == (nodes[0].id,)


def test_optional_card_state_persists_across_rebuilds(
    qtbot, viewer, synth_centroided
):
    """Going Back to WorkflowPage and re-entering must preserve the user's enable toggle."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)

    hot_card = next(c for c in page._cards if c._node.op_name == "hot_pixel_filter")  # noqa: SLF001
    hot_card._enable_checkbox.setChecked(True)  # noqa: SLF001

    # Simulate Back navigation re-entering the page.
    page.initializePage()
    hot_card2 = next(c for c in page._cards if c._node.op_name == "hot_pixel_filter")  # noqa: SLF001
    assert hot_card2.is_enabled(), (
        "the user's choice to enable hot-pixel filter must survive page rebuilds"
    )


def test_optional_card_form_is_disabled_when_unchecked(qtbot, viewer, synth_centroided):
    """When the optional card's checkbox is unchecked, the param form widgets are greyed out."""
    wiz, page = _open_workflow_page(viewer, synth_centroided)
    qtbot.addWidget(wiz)

    hot_card = next(c for c in page._cards if c._node.op_name == "hot_pixel_filter")  # noqa: SLF001
    # Default is disabled — form widget should be disabled too.
    assert not hot_card._form_widget.isEnabled()  # noqa: SLF001
    hot_card._enable_checkbox.setChecked(True)  # noqa: SLF001
    assert hot_card._form_widget.isEnabled()  # noqa: SLF001
