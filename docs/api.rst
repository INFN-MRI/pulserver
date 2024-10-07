API References
==============

Core
----
Core sequence representation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.Sequence
   pulserver.SequenceParams
   pulserver.Opts 

Blocks
------
Subroutines for the generation of sequence blocks, e.g., 
preparation modules, rf pulses, phase encoding, readout.

RF Pulses
^^^^^^^^^
Non-adiabatic RF pulses blocks, including both the RF events
and (for spatially-selective pulses), the accompanying gradient event.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulserver.blocks.make_hard_pulse
   pulserver.blocks.make_slr_pulse
   pulserver.blocks.make_spsp_pulse
   