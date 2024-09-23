"""3D Spoiled Gradient Echo sequence."""

__all__ = []

from typing import Sequence


import numpy as np
import pypulseq as pp


from pulseforge import PulseqBlock


def design_spgr3D(
    fov: Sequence[float],
    mtx: Sequence[int],
    alpha: float,
    TR: float,
    max_grad: float,
    max_slew: float,
    grad_raster_time: float,
):
    # RF specs
    rf_spoiling_inc = 117.0  # RF spoiling increment

    # initialize system limits
    system = pp.Opts(
        max_grad=max_grad,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        grad_raster_time=grad_raster_time,
    )

    # initialize sequence
    seq = PulseqBlock(system=system)  # TODO: rename as Sequence?
    fov, slab_thickness = fov[0] * 1e-3, fov[1] * 1e-3  # in-plane FOV, slab thickness
    Nx, Ny, Nz = mtx[0], mtx[0], mtx[1]  # in-plane resolution, slice thickness

    # initialize event events
    # RF pulse
    rf, gss, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(alpha),
        duration=3e-3,
        slice_thickness=slab_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
    )
    gss_reph = pp.make_trapezoid(
        channel="z", area=-gss.area / 2, duration=1e-3, system=system
    )
    seq.register_event(name="excitation", rf=rf, gz=gss)
    seq.register_event(name="slab_rephasing", gz=gss_reph)

    # readout
    delta_kx, delta_ky, delta_kz = 1 / fov, 1 / fov, 1 / slab_thickness
    gread = pp.make_trapezoid(
        channel="x", flat_area=Nx * delta_kx, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(
        num_samples=Nx, duration=gread.flat_time, delay=gread.rise_time, system=system
    )
    gxprew = pp.make_trapezoid(
        channel="x", area=-gread.area / 2, duration=1e-3, system=system
    )  # PREwinder / REWinder
    seq.register_event(name="gxprew", gx=gxprew)
    seq.register_event(name="readout", gx=gread, adc=adc)

    # phase encoding
    gyphase = pp.make_trapezoid(channel="y", area=-delta_ky * Ny, system=system)
    gzphase = pp.make_trapezoid(channel="z", area=-delta_kz * Nz, system=system)
    seq.register_event("gphase", gy=gyphase, gz=gzphase)

    # crusher gradient
    gzspoil = pp.make_trapezoid(channel="z", area=4 / slab_thickness, system=system)
    seq.register_event("gspoil", gz=gzspoil)

    # phase encoding plan TODO: helper routine
    pey_steps = ((np.arange(Ny)) - Ny / 2) / Ny * 2
    pez_steps = ((np.arange(Nz)) - Nz / 2) / Nz * 2

    # construct sequence
