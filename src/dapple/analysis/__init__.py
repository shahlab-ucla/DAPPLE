"""Post-harmonization spatial and developmental-pattern analysis.

The analysis package deliberately sits beside, rather than inside, the processing
operator DAG.  Its functions consume a harmonized :class:`~dapple.data.dataset.PeakMatrix`
and return immutable result objects with table/export helpers; they never mutate or
filter the input dataset.
"""

from dapple.analysis.developmental import (
    analyze_axis_profiles,
    analyze_roi_enrichment,
    benjamini_hochberg,
)
from dapple.analysis.export import ExportedAnalysis, export_analysis_result
from dapple.analysis.models import AxisProfileResult, RoiEnrichmentResult
from dapple.analysis.spatial import (
    AxisProjection,
    DirectedAxis,
    OverlapPolicy,
    RoiMasks,
    coordinate_fingerprint,
    project_to_axis,
    rasterize_rois,
)

__all__ = [
    "AxisProfileResult",
    "AxisProjection",
    "DirectedAxis",
    "ExportedAnalysis",
    "OverlapPolicy",
    "RoiEnrichmentResult",
    "RoiMasks",
    "analyze_axis_profiles",
    "analyze_roi_enrichment",
    "benjamini_hochberg",
    "coordinate_fingerprint",
    "export_analysis_result",
    "project_to_axis",
    "rasterize_rois",
]
