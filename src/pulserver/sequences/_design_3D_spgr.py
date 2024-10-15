# """3D Spoiled Gradient Echo sequence."""

# __all__ = ["SPGR3D"]

# from collections.abc import Iterable


# import numpy as np
# import pypulseq as pp


# from pulserver import Sequence
# from pulserver.plan import RfPhaseCycle


# def SPGR3D(
#     fov: Iterable[float],
#     npix: Iterable[int],
#     alpha: float,
#     max_grad: float,
#     max_slew: float,
#     raster_time: float,
#     seqformat: str | bool = "bytes",
# ):
#     """
#     Generate a 3D Spoiled Gradient Recalled Echo (SPGR) pulse sequence.

#     This function designs a 3D SPGR sequence based on the provided field of view (FOV), matrix size,
#     flip angle, and hardware constraints such as maximum gradient amplitude and slew rate. The output
#     can be formatted in different sequence file formats if specified.

#     Parameters
#     ----------
#     fov : Iterable[float]
#         Field of view along each spatial dimension [fov_plane, fov_z] in mm.
#         If scalar, assume cubic fov.
#     npix : Iterable[int]
#         Number of voxels along each spatial dimension [plane_mtx, nz] (matrix size).
#         If scalar, assume cubic matrix size.
#     alpha : float
#         Flip angle in degrees.
#     max_grad : float
#         Maximum gradient amplitude in mT/m.
#     max_slew : float
#         Maximum gradient slew rate in T/m/s.
#     raster_time : float
#         Waveform raster time in seconds (the time between successive gradient samples).
#     seqformat : str or bool, optional
#         Output sequence format. If a string is provided, it specifies the desired output format (e.g., 'pulseq', 'bytes').
#         If False, the sequence is returned as an internal object. Default is False.

#     Returns
#     -------
#     seq : object or dict
#         The generated SPGR sequence. If `seqformat` is a string, the sequence is returned in the specified format.
#         If `seqformat` is False, the sequence is returned as an internal representation.

#     Notes
#     -----
#     - This function is designed to work within the constraints of MRI scanners, taking into account the physical limits
#       on gradient amplitude and slew rates.
#     - The flip angle (`alpha`) controls the excitation of spins and directly impacts the signal-to-noise ratio (SNR) and contrast.

#     Examples
#     --------
#     Generate a 3D SPGR sequence for a 256x256x128 matrix with a 240x240x120 mm FOV, a 15-degree flip angle, and hardware limits:

#     >>> from pulseforge import SPGR3D
#     >>> SPGR3D([240, 120], [256, 128], 15, 0.04, 150, 4e-6)

#     Generate the same sequence and export it in bytes format:

#     >>> SPGR3D([240, 120], [256, 128], 15, 0.04, 150, 4e-6, seqformat='bytes')

#     """
#     # RF specs
#     rf_spoiling_inc = 117.0  # RF spoiling increment

#     # initialize system limits
#     system = pp.Opts(
#         max_grad=max_grad,
#         grad_unit="mT/m",
#         max_slew=max_slew,
#         slew_unit="T/m/s",
#         grad_raster_time=raster_time,
#         rf_raster_time=raster_time,
#     )

#     # initialize sequence
#     seq = Sequence(system=system, format=seqformat)

#     # initialize prescription
#     if np.isscalar(fov):
#         fov, slab_thickness = fov * 1e-3, fov * 1e-3  # isotropic
#     else:
#         fov, slab_thickness = (
#             fov[0] * 1e-3,
#             fov[1] * 1e-3,
#         )  # in-plane FOV, slab thickness

#     if np.isscalar(npix):
#         Nx, Ny, Nz = npix, npix, npix  # in-plane resolution, slice thickness
#     else:
#         Nx, Ny, Nz = npix[0], npix[0], npix[1]  # in-plane resolution, slice thickness

#     # initialize event events
#     # RF pulse
#     rf, gss, _ = pp.make_sinc_pulse(
#         flip_angle=np.deg2rad(alpha),
#         duration=3e-3,
#         slice_thickness=slab_thickness,
#         apodization=0.42,
#         time_bw_product=4,
#         system=system,
#         return_gz=True,
#     )
#     gss_reph = pp.make_trapezoid(
#         channel="z", area=-gss.area / 2, duration=1e-3, system=system
#     )
#     seq.register_block(name="excitation", rf=rf, gz=gss)
#     seq.register_block(name="slab_rephasing", gz=gss_reph)

#     # readout
#     delta_kx, delta_ky, delta_kz = 1 / fov, 1 / fov, 1 / slab_thickness
#     g_read = pp.make_trapezoid(
#         channel="x", flat_area=Nx * delta_kx, flat_time=3.2e-3, system=system
#     )
#     adc = pp.make_adc(
#         num_samples=Nx, duration=g_read.flat_time, delay=g_read.rise_time, system=system
#     )
#     seq.register_block("dummy_readout", gx=g_read)
#     seq.register_block("readout", gx=g_read, adc=adc)

#     # phase encoding
#     gx_phase = pp.make_trapezoid(
#         channel="x", area=-g_read.area / 2, duration=1e-3, system=system
#     )
#     gy_phase = pp.make_trapezoid(channel="y", area=delta_ky * Ny, system=system)
#     gz_phase = pp.make_trapezoid(channel="z", area=delta_kz * Nz, system=system)
#     seq.register_block("g_phase", gx=gx_phase, gy=gy_phase, gz=gz_phase)

#     # crusher gradient
#     gz_spoil = pp.make_trapezoid(channel="z", area=32 / slab_thickness, system=system)
#     seq.register_block("g_spoil", gz=gz_spoil)

#     # phase encoding plan TODO: helper routine
#     pey_steps = ((np.arange(Ny)) - (Ny / 2)) / Ny
#     pez_steps = ((np.arange(Nz)) - (Nz / 2)) / Nz
#     encoding_plan = np.meshgrid(pey_steps, pez_steps, indexing="xy")
#     encoding_plan = [enc.ravel() for enc in encoding_plan]

#     # scan duration
#     dummy_scans = Ny
#     imaging_scans = Ny * Nz

#     # generate rf phases
#     rf_phases = RfPhaseCycle(
#         num_pulses=dummy_scans + imaging_scans, phase_increment=rf_spoiling_inc
#     )

#     # construct sequence
#     seq.section(name="ss_prep")
#     for n in range(dummy_scans):
#         rf_phase = rf_phases()
#         seq.add_block("excitation")
#         seq.add_block("slab_rephasing")
#         seq.add_block("g_phase", gy_amp=0.0, gz_amp=0.0)
#         seq.add_block("dummy_readout", rf_phase=rf_phase)
#         seq.add_block("g_phase", gy_amp=0.0, gz_amp=0.0)
#         seq.add_block("g_spoil")

#     seq.section(name="scan_loop")
#     for n in range(imaging_scans):
#         rf_phase = rf_phases()
#         seq.add_block("excitation")
#         seq.add_block("slab_rephasing")
#         seq.add_block("g_phase", gy_amp=encoding_plan[0][n], gz_amp=encoding_plan[1][n])
#         seq.add_block("readout", rf_phase=rf_phase, adc_phase=rf_phase)
#         seq.add_block(
#             "g_phase", gy_amp=-encoding_plan[0][n], gz_amp=-encoding_plan[1][n]
#         )
#         seq.add_block("g_spoil")

#     return seq.export()
