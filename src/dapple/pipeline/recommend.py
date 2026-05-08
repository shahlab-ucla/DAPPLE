"""recommend_pipeline: suggest a default operator chain from ExperimentParams.

Builds an instrument-aware default chain and returns a topologically ordered
Pipeline. Future operators (recalibration, spatial filtering, profile-mode peak
picking) will be wired in once they exist; the recommendation is intentionally
conservative — nothing that imputes, nothing parametric without an empirical
diagnostic.
"""

from __future__ import annotations

from dapple.data.metadata import ExperimentParams
from dapple.ops.base import REGISTRY
from dapple.pipeline.pipeline import Node, Pipeline, detect_library_versions


def recommend_pipeline(ep: ExperimentParams, *, rng_seed: int = 0) -> Pipeline:
    """Build the default pipeline for the supplied experiment.

    Always:
        detect_reference_ions → empirical_tolerance_from_reference_ions →
        recalibrate (TOF / Q-TOF only) → median_normalize → peak_pick →
        kde_consensus_alignment → spatial_filter (tissue only).

    Picker selection: ``cwt_peak_pick`` for profile data, ``snr_peak_pick`` for
    centroided. Recalibration is inserted for TOF / Q-TOF families where mass
    drift is the dominant residual; Orbitrap / FT-ICR data is locked enough that
    recalibration adds noise. Spatial filtering is inserted for tissue samples
    (``sample_type='tissue'``) where the noise-channel rejection is most useful;
    cell culture and other dispersed samples don't have the spatial coherence
    that Moran's I assumes.
    """
    nodes: list[Node] = []

    ref_op = REGISTRY.get("detect_reference_ions")()
    nodes.append(
        Node(
            id="ref",
            op_name="detect_reference_ions",
            params=ref_op.default_params(ep),
            upstream=(),
        )
    )

    tol_op = REGISTRY.get("empirical_tolerance_from_reference_ions")()
    nodes.append(
        Node(
            id="tol",
            op_name="empirical_tolerance_from_reference_ions",
            params=tol_op.default_params(ep),
            upstream=("ref",),
        )
    )

    last_id = "tol"
    if ep.instrument_family in {"tof_axial", "tof_reflectron", "qtof"}:
        recal_op = REGISTRY.get("msiwarp_recalibrate")()
        nodes.append(
            Node(
                id="recal",
                op_name="msiwarp_recalibrate",
                params=recal_op.default_params(ep),
                upstream=("tol",),
            )
        )
        last_id = "recal"

    norm_op = REGISTRY.get("median_normalize")()
    nodes.append(
        Node(
            id="norm",
            op_name="median_normalize",
            params=norm_op.default_params(ep),
            upstream=(last_id,),
        )
    )

    pick_op_name = "cwt_peak_pick" if ep.profile_or_centroided == "profile" else "snr_peak_pick"
    pick_op = REGISTRY.get(pick_op_name)()
    nodes.append(
        Node(
            id="pick",
            op_name=pick_op_name,
            params=pick_op.default_params(ep),
            upstream=("norm",),
        )
    )

    cons_op = REGISTRY.get("kde_consensus_alignment")()
    nodes.append(
        Node(
            id="consensus",
            op_name="kde_consensus_alignment",
            params=cons_op.default_params(ep),
            upstream=("pick",),
        )
    )

    if ep.sample_type == "tissue":
        spatial_op = REGISTRY.get("morans_i_permutation")()
        nodes.append(
            Node(
                id="spatial",
                op_name="morans_i_permutation",
                params=spatial_op.default_params(ep),
                upstream=("consensus",),
            )
        )

    return Pipeline(
        nodes=tuple(nodes),
        rng_seed=rng_seed,
        library_versions=detect_library_versions(),
    )
