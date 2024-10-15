"""2D Spoiled Gradient Echo sequence."""

__all__ = ["design_2D_spgr"]

from collections.abc import Iterable


import numpy as np
import pypulseq as pp


from .._core import Sequence
from .. import blocks
from .. import plan


def design_2D_spgr(
    fov: Iterable[float],
    slice_thickness: float,
    matrix_size: Iterable[int],
    n_slices: int,
    flip_angle: float,
    R: int,
    PF: float,
    rectime: float,
    max_grad: float,
    max_slew: float,
    raster_time: float,
    slice_spacing: float = 0.0,
    platform: str = "pulseq",
):
    """
    Generate a 2D Spoiled Gradient Recalled Echo (SPGR) pulse sequence.

    This function designs a 2D SPGR sequence based on the provided field of view (FOV), matrix size,
    number of slices, slice thickness and spacing, flip angle, recovery time,
    and hardware constraints such as maximum gradient amplitude and slew rate.
    The output can be formatted in different sequence file formats if specified.

    Parameters
    ----------
    fov : Iterable[float]
        Field of view along each spatial dimension (fov_x, fov_y) in mm.
        If scalar, assume squared fov.
    slice_thickness : float)
        Slice thickness in mm.
    matrix_size  : Iterable[int]
        Number of voxels along each spatial dimension (nx, ny) (matrix size).
        If scalar, assume squared matrix size.
    n_slices : int
        Number of slices.
    flip_angle : float
        Flip angle in degrees.
    rectime : float
        Recovery time after spoiling in ms.
    max_grad : float
        Maximum gradient amplitude in mT/m.
    max_slew : float
        Maximum gradient slew rate in T/m/s.
    raster_time : float
        Waveform raster time in seconds (the time between successive gradient samples).
    slice_spacing : float, optional
        Additional slice spacing in mm. The default is 0.0 (contiguous slices).
    seqformat : str or bool, optional
        Output sequence format. If a string is provided, it specifies the desired output format (e.g., 'pulseq', 'bytes').
        If False, the sequence is returned as an internal object. Default is False.

    Returns
    -------
    seq : object or dict
        The generated SPGR sequence. If `seqformat` is a string, the sequence is returned in the specified format.
        If `seqformat` is False, the sequence is returned as an internal representation.

    Notes
    -----
    - This function is designed to work within the constraints of MRI scanners, taking into account the physical limits
      on gradient amplitude and slew rates.
    - The flip angle (`flip_angle`) controls the excitation of spins and directly impacts the signal-to-noise ratio (SNR) and contrast.

    Examples
    --------
    Generate a 2D SPGR sequence for a single 5 mm thick slice and 240x240 mm FOV, 256x256 matrix size,
    15-degree flip angle, 5s recovery time and hardware limits 4mT/m, 150T/m/s, 4e-6 s raster time as:

    >>> from pulseforge import SPGR2D
    >>> design_2D_spgr(240.0, 5.0, 256, 1, 15.0, 5.0, 40, 150, 4e-6)

    Generate the same sequence and export it in GEHC format:

    >>> design_2D_spgr(240.0, 5.0, 256, 1, 15.0, 5.0, 40, 150, 4e-6, platform='gehc')

    """
    # Sequence Parameters
    # -------------------
    rf_spoiling_inc = 117.0  # RF spoiling increment

    # initialize prescription
    slice_spacing += slice_thickness

    if np.isscalar(fov):
        FOVx, FOVy = fov, fov
    else:
        FOVx, FOVy = fov

    if np.isscalar(matrix_size):
        Nx, Ny = matrix_size, matrix_size
    else:
        Nx, Ny = matrix_size[0], matrix_size[1]

    # initialize system limits
    system_limits = pp.Opts(
        max_grad=max_grad,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        grad_raster_time=raster_time,
        rf_raster_time=raster_time,
    )

    # initialize sequence object
    seq = Sequence(system=system_limits, platform=platform)

    # Define Blocks
    # -------------
    # create excitation and slice rephasing blocks
    exc_block, slice_reph_block = blocks.make_slr_pulse(
        system_limits, flip_angle, slice_thickness
    )

    # create phase encoding gradient, readout pre-/re-winder and readout/adc blocks:
    phase_enc = blocks.make_phase_encoding("y", system_limits, FOVy, Ny)
    readout_block, readout_prewind_block = blocks.make_line_readout(
        system_limits, FOVx, Nx
    )

    # create combined phase encoding + readout prewinder block
    phase_enc_block = {"gy": phase_enc, **readout_prewind_block}

    # create spoiling block
    spoil_block = {
        "gz": blocks.make_spoiler_gradient(
            "z", system_limits, ncycles=4, voxel_size=slice_thickness
        )
    }

    # register parent blocks
    seq.register_block(name="excitation", **exc_block)
    seq.register_block(name="slice_rephasing", **slice_reph_block)
    seq.register_block(name="readout", **readout_block)
    seq.register_block(name="dummy_readout", gx=readout_block["gx"])
    seq.register_block(name="phase_encoding", **phase_enc_block)
    seq.register_block(name="spoiling", **spoil_block)

    # Define sequence plan
    # --------------------
    # scan duration
    dummy_scans = 10
    calib_scans = 10
    imaging_scans = Ny * n_slices

    # generate rf phase schedule
    rf_phases = plan.RfPhaseCycle(
        num_pulses=dummy_scans + imaging_scans + calib_scans,
        phase_increment=rf_spoiling_inc,
    )

    # create Gy and RF frequency offset schedule to achieve the requested FOV, in-plane resolution and number of slices
    encoding_plan, _ = plan.cartesian2D(
        g_slice_select=exc_block["gz"],
        slice_thickness=slice_thickness,
        n_slices=n_slices,
        ny=Ny,
        dummy_shots=calib_scans + dummy_scans,
    )

    # Set up scan loop
    # ----------------
    # Steady state preparation
    seq.section(name="ss_prep")
    for n in range(dummy_scans):

        # get dynamic sequence parameters
        rf_phase = rf_phases()
        encoding, _ = encoding_plan()

        # update sequence loop
        seq.add_block("excitation", rf_phase=rf_phase, rf_freq=encoding.rf_freq)
        seq.add_block("slice_rephasing")
        seq.add_block("phase_encoding", gy_amp=encoding.gy_amp)
        seq.add_block("dummy_readout")
        seq.add_block("phase_encoding", gy_amp=-encoding.gy_amp)
        seq.add_block("spoiling")

    # Actual sequence
    seq.section(name="scan_loop")
    for n in range(imaging_scans + calib_scans):

        # get dynamic sequence parameters
        rf_phase = rf_phases()
        encoding, _ = encoding_plan()

        # update sequence loop
        seq.add_block("excitation", rf_phase=rf_phase, rf_freq=encoding.rf_freq)
        seq.add_block("slice_rephasing")
        seq.add_block("phase_encoding", gy_amp=encoding.gy_amp)
        seq.add_block("readout", adc_phase=rf_phase)
        seq.add_block("phase_encoding", gy_amp=-encoding.gy_amp)
        seq.add_block("spoiling")

    # build the sequence
    return seq.build()
