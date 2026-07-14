"""Napari widgets: wizard, channels panel, spectrum panel, preview, ROI."""

from dapple.widgets._session import MsiSession, adopt_dataset_from_viewer, default_session
from dapple.widgets.channels import ChannelsPanel
from dapple.widgets.developmental import DevelopmentalAnalysisWidget
from dapple.widgets.preview import PreviewWidget
from dapple.widgets.roi import RoiWidget
from dapple.widgets.spectrum import SpectrumPanel
from dapple.widgets.threshold_explorer import ThresholdExplorerPanel
from dapple.widgets.wizard import WizardWidget

# `HyperspectralBrowser` was the original per-channel viewer. It's been replaced by
# `ChannelsPanel`, which presents both summary projections and consensus m/z channels
# in a single LUT-aware table. We keep an alias for any external code that still
# imports the old name; new code should use ``ChannelsPanel`` directly.
HyperspectralBrowser = ChannelsPanel

__all__ = [
    "ChannelsPanel",
    "DevelopmentalAnalysisWidget",
    "HyperspectralBrowser",
    "MsiSession",
    "PreviewWidget",
    "RoiWidget",
    "SpectrumPanel",
    "ThresholdExplorerPanel",
    "WizardWidget",
    "adopt_dataset_from_viewer",
    "default_session",
]
