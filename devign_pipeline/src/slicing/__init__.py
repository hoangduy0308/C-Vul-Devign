"""Code slicing module for vulnerability detection"""

from .slicer import (
    SliceType,
    SliceConfig,
    CodeSlice,
    CodeSlicer,
    slice_batch,
)

from .pdg_slicer import (
    PDGSliceType,
    PDGSliceConfig,
    PDGSliceResult,
    PDGSlicer,
    pdg_slice,
)

__all__ = [
    # Window-based slicer (legacy)
    'SliceType',
    'SliceConfig', 
    'CodeSlice',
    'CodeSlicer',
    'slice_batch',
    # PDG-based slicer (recommended)
    'PDGSliceType',
    'PDGSliceConfig',
    'PDGSliceResult',
    'PDGSlicer',
    'pdg_slice',
]
