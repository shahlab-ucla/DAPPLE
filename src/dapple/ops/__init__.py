"""Operator library for MSI processing."""

from dapple.ops.background import BackgroundSubtract, BackgroundSubtractParams
from dapple.ops.base import REGISTRY, Diagnostic, OpParams, OpResult, Operator, register
from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
from dapple.ops.dbscan_consensus import DbscanConsensusAlignment, DbscanConsensusParams
from dapple.ops.hot_pixel import HotPixelFilter, HotPixelParams
from dapple.ops.normalize import (
    MedianNormalize,
    NormalizeParams,
    ReferenceIonNormalize,
    ReferenceIonNormalizeParams,
    TicNormalize,
)
from dapple.ops.peak_pick import (
    CwtPeakPick,
    CwtPeakPickParams,
    SnrPeakPick,
    SnrPeakPickParams,
)
from dapple.ops.prevalence_filter import (
    PrevalenceFdrFilter,
    PrevalenceFdrParams,
)
from dapple.ops.recalibrate import (
    LockMassRecalibrate,
    LockMassRecalibrateParams,
    MsiwarpRecalibrate,
    MsiwarpRecalibrateParams,
)
from dapple.ops.reference_ions import (
    DetectReferenceIons,
    ReferenceIonsParams,
    ReferenceSet,
)
from dapple.ops.spatial_filter import MoransIParams, MoransIPermutation
from dapple.ops.tolerance import (
    EmpiricalToleranceFromReferenceIons,
    EmpiricalToleranceParams,
    ToleranceCurve,
)

__all__ = [
    "REGISTRY",
    "BackgroundSubtract",
    "BackgroundSubtractParams",
    "CwtPeakPick",
    "CwtPeakPickParams",
    "DbscanConsensusAlignment",
    "DbscanConsensusParams",
    "DetectReferenceIons",
    "Diagnostic",
    "EmpiricalToleranceFromReferenceIons",
    "EmpiricalToleranceParams",
    "HotPixelFilter",
    "HotPixelParams",
    "KdeConsensusAlignment",
    "KdeConsensusParams",
    "LockMassRecalibrate",
    "LockMassRecalibrateParams",
    "MedianNormalize",
    "MoransIParams",
    "MoransIPermutation",
    "MsiwarpRecalibrate",
    "MsiwarpRecalibrateParams",
    "NormalizeParams",
    "OpParams",
    "OpResult",
    "Operator",
    "PrevalenceFdrFilter",
    "PrevalenceFdrParams",
    "ReferenceIonNormalize",
    "ReferenceIonNormalizeParams",
    "ReferenceIonsParams",
    "ReferenceSet",
    "SnrPeakPick",
    "SnrPeakPickParams",
    "TicNormalize",
    "ToleranceCurve",
    "register",
]
