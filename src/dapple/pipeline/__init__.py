"""Pipeline orchestration: DAG, runner, recommend_pipeline."""

from dapple.pipeline.diag_format import (
    COHORT_RUBRICS,
    format_diagnostics,
    format_flat_summary,
    format_node_diagnostics,
)
from dapple.pipeline.pipeline import Node, Pipeline, detect_library_versions
from dapple.pipeline.recommend import recommend_pipeline
from dapple.pipeline.runner import PipelineRunner, RunResult

__all__ = [
    "COHORT_RUBRICS",
    "Node",
    "Pipeline",
    "PipelineRunner",
    "RunResult",
    "detect_library_versions",
    "format_diagnostics",
    "format_flat_summary",
    "format_node_diagnostics",
    "recommend_pipeline",
]
