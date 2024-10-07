"""Cartesian readout creation subroutines."""

__all__ = ["CartesianReadout2D", "CartesianReadout3D"]

import numpy as np

import pypulseq as pp

from .._block import PulseqBlock


def make_line_readout(
        system: pp.Opts, 
        fov: float, 
        npix: int, 
        osf: float = 1.0, 
        adc_duration: float = 3.0,
        ):
        """
        Prepare line readout for Cartesian, EPI or Radial imaging.

        Parameters
        ----------
        system : pypulseq.Opts
            System limits.
        fov : float | tuple[float]
            Field of view ``(FOVx, FOVy)`` in ``[mm]``.
            If it is a scalar, assume isotropic FOV (``fov=(fov, fov)``).
        npix : int | tuple[int]
            Matrix size ``(nx, ny)``.
            If it is a scalar, assume square matrix (``npix=(npix, npix)``).
        osf : float, optional
            Readout oversampling factor. The default is ``1.0``.
        adc_duration : float, optional
            ADC window duration in ``[ms]``. The default is '`3.0``.

        """
        # store number of echoes and flyback
        self.nechoes = nechoes
        self.flyback = flyback

        # parse prescription
        if np.isscalar(fov):
            fov = [fov, fov]
        if np.isscalar(npix):
            npix = [npix, npix]

        # unpack fov and matrix
        FOVx, FOVy = fov
        nx, ny = npix

        # convert ms -> s
        adc_duration *= 1e-3

        # apply oversampling
        nx = osf * nx

        # k space density
        dkx, dky = 1 / FOVx, 1 / FOVy

        # frequency encoding gradients
        gread = pp.make_trapezoid(
            "x", flat_area=nx * dkx, flat_time=adc_duration, system=sys
        )
        adc = pp.make_adc(
            num_samples=nx, duration=gread.flat_time, delay=gread.rise_time, system=sys
        )
        gxphase = pp.make_trapezoid("x", area=-gread.area / 2, system=sys)

        # flyback gradient for multiecho
        if nechoes > 1 and flyback:
            gflyback = pp.make_trapezoid("x", area=-gread.area, system=sys)

        # phase encoding gradient
        gyphase = pp.make_trapezoid(channel="y", area=dky * ny, system=sys)

        # set list of blocks
        self.add_block("phase_encoding", gx=gxphase, gy=gyphase)
        self.add_block("readout", gx=gread, adc=adc)
        self.add_block("dummy_readout", gx=gread)

        if flyback:
            self.add_block("flyback", gx=gflyback)

    def __call__(self, scanloop, yscale, recphase=0.0, dummy=False):
        """
        Update scanloop with a (multi-echo) 2D Cartesian readout.

        The module consists of:

            1. Phase encoding (along x, y axes) for prewind.
            2. Frequency encoding with signal acquisition (except for dummy scans)
            3. Flyback (optional) + repeated application of frequency encodings.
            4. Phase encoding (along x, y axes) for rewind.

        Parameters
        ----------
        scanloop : SimpleNamespace
            Structure containing dynamic scan information.
        yscale : float
            Scaling of y-axis phase encoding, with +-1.0 being +-ky_max.
        recphase : float, optional
            ADC phase offset for signal demodulation. The default is ``0.0``.
        dummy : bool, optional
            If ``True``, do not acquire data during readout. The default is ``False``.

        """
        # add prewinder
        scanloop.update("phase_encoding", gxamp=1.0, gyamp=yscale)

        # add (multiecho) readout
        if dummy:
            for e in range(self.nechoes):
                if self.flyback:
                    xscale = 1.0
                else:
                    xscale = -(1.0 ** (e + 2))
                scanloop.update("dummy_readout", gxamp=xscale)
                if self.flyback:
                    scanloop.update("flyback")
        else:
            for e in range(self.nechoes):
                if self.flyback:
                    xscale = 1.0
                else:
                    xscale = -(1.0 ** (e + 2))
                scanloop.update("readout", gxamp=xscale, recphase=recphase)
                if self.flyback:
                    scanloop.update("flyback")

        # add rewinder
        scanloop.update("phase_encoding", gxamp=-1.0, gyamp=-yscale)


class CartesianReadout3D(CartesianReadout2D):
    """
    Wrapper class for PyPulseq 3D Cartesian readout module.
    """

    def __init__(
        self, sys, fov, npix, osf=1.0, adc_duration=3.0, nechoes=1, flyback=False
    ):
        """
        Prepare Cartesian readout for 2D imaging.

        Parameters
        ----------
        sys : pypulseq.Opts
            System limits.
        fov : float | tuple[float]
            Field of view ``(FOVx, FOVy, FOVz)`` in ``[mm]``.
            If it is a scalar, assume isotropic FOV (``fov=(fov, fov, fov)``).
        npix : int | tuple[int]
            Matrix size ``(nx, ny, nz)``.
            If it is a scalar, assume cubic matrix (``npix=(npix, npix, npix)``).
        osf : float, optional
            Readout oversampling factor. The default is ``1.0``.
        adc_duration : float, optional
            ADC window duration in ``[ms]``. The default is '`3.0``.
        nechoes : int, optional
            Number of echoes. The default is ``1``.
        flyback : bool, optional
            Flyback (monopolar) echoes (``True``) or bipolar echoes (``False``).
            Ignored for single-echo acquisitions (``nechoes=1``).
            The default is ``False``.

        """
        # store number of echoes and flyback
        self.nechoes = nechoes
        self.flyback = flyback

        # parse prescription
        if np.isscalar(fov):
            fov = [fov, fov, fov]
        if np.isscalar(npix):
            npix = [npix, npix, npix]

        # unpack fov and matrix
        FOVx, FOVy, FOVz = fov
        nx, ny, nz = npix

        # convert ms -> s
        adc_duration *= 1e-3

        # apply oversampling
        nx = osf * nx

        # k space density
        dkx, dky, dkz = 1 / FOVx, 1 / FOVy, 1 / FOVz

        # frequency encoding gradients
        gread = pp.make_trapezoid(
            "x", flat_area=nx * dkx, flat_time=adc_duration, system=sys
        )
        adc = pp.make_adc(
            num_samples=nx, duration=gread.flat_time, delay=gread.rise_time, system=sys
        )
        gxphase = pp.make_trapezoid("x", area=-gread.area / 2, system=sys)

        # flyback gradient for multiecho
        if nechoes > 1 and flyback:
            gflyback = pp.make_trapezoid("x", area=-gread.area, system=sys)

        # phase encoding gradient
        gyphase = pp.make_trapezoid(channel="y", area=dky * ny, system=sys)
        gzphase = pp.make_trapezoid(channel="z", area=dkz * ny, system=sys)

        # set list of blocks
        self.add_block("phase_encoding", gx=gxphase, gy=gyphase, gz=gzphase)
        self.add_block("readout", gx=gread, adc=adc)
        self.add_block("dummy_readout", gx=gread)

        if flyback:
            self.add_block("flyback", gx=gflyback)

    def __call__(self, scanloop, yscale, zscale, recphase=0.0, dummy=False):
        """
        Update scanloop with a (multi-echo) 3D Cartesian readout.

        The module consists of:

            1. Phase encoding (along x, y and z axes) for prewind.
            2. Frequency encoding with signal acquisition (except for dummy scans)
            3. Flyback (optional) + repeated application of frequency encodings.
            4. Phase encoding (along x, y and z axes) for rewind.

        Parameters
        ----------
        scanloop : SimpleNamespace
            Structure containing dynamic scan information.
        yscale : float
            Scaling of y-axis phase encoding, with +-1.0 being +-ky_max.
        zscale : float
            Scaling of z-axis phase encoding, with +-1.0 being +-kz_max.
        recphase : float, optional
            ADC phase offset for signal demodulation. The default is ``0.0``.
        dummy : bool, optional
            If ``True``, do not acquire data during readout. The default is ``False``.

        """
        # add prewinder
        scanloop.update("phase_encoding", gxamp=1.0, gyamp=yscale, gzamp=zscale)

        # add (multiecho) readout
        if dummy:
            for e in range(self.nechoes):
                if self.flyback:
                    xscale = 1.0
                else:
                    xscale = -(1.0 ** (e + 2))
                scanloop.update("dummy_readout", gxamp=xscale)
                if self.flyback:
                    scanloop.update("flyback")
        else:
            for e in range(self.nechoes):
                if self.flyback:
                    xscale = 1.0
                else:
                    xscale = -(1.0 ** (e + 2))
                scanloop.update("readout", gxamp=xscale, recphase=recphase)
                if self.flyback:
                    scanloop.update("flyback")

        # add rewinder
        scanloop.update("phase_encoding", gxamp=-1.0, gyamp=-yscale, gzamp=-zscale)
