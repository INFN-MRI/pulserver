"""PyPulseq sub-package.

This sub-package contains PyPulseq-based sub-routines
to generate sequence events (e.g., RF pulses, gradient waveforms).
"""

__all__ = []

# %%
# Adiabatic pulses
# ================

# %%
# RF pulses
# =========
# from pypulseq import make_block_pulse  # Non-selective
# from pypulseq import make_slr as make_slr_pulse  # Frequency / Slice selective
from . import _make_spsp_pulse

from ._make_spsp_pulse import make_spsp_pulse

__all__ = ["make_spsp_pulse"]
