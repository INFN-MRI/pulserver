"""Spiral readout creation subroutines."""

__all__ = ["SpiralReadout", "StackOfSpiralsReadout"]

import numpy as np

import pypulseq as pp
import pulpy.grad as pg

from .._block import PulseqBlock


class SpiralReadout(PulseqBlock):
    pass


class StackOfSpiralsReadout(PulseqBlock):
    pass
