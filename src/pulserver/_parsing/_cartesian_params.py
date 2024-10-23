"""Single Echo Cartesian Imaging parameter parsing."""

__all__ = ["Cartesian2DParams", "Cartesian3DParams"]

from ._base import BaseParams


class Cartesian2DParams(BaseParams):
    """
    Parse parameters for single-echo 2D Cartesian imaging.

    This is designed to convert data parsed by ParamsParser
    to a format suitable for single-echo 2D Cartesian design routines.

    """

    def __init__(
        self,
        FOVx: float | None = None,
        FOVy: float | None = None,
        Nx: int | None = None,
        Ny: int | None = None,
        Nslices: int | None = None,
        slice_thickness: float | None = None,
        slice_spacing: float | None = 0.0,
        Rplane: float | None = 1,
        PFplane: float | None = 1.0,
        TE: float | None = 0.0,
        TR: float | None = 0.0,
        flip: float | None = None,
        dwell: float | None = 4e-6,
        raster: float | None = 2e-6,
        gmax: float | None = None,
        smax: float | None = None,
        b0_field: float | None = None,
        rf_dead_time: float | None = 100e-6,
        rf_ringdown_time: float | None = 60e-6,
        adc_dead_time: float | None = 40e-6,
        psd_rf_wait: float | None = 0.0,
        *args,
        **kwargs,
    ):  # noqa
        # Build FOV
        if FOVx is None:
            raise ValueError("Please provide FOVx")
        if FOVy is None:
            self.fov = (FOVx, FOVy)
        else:
            self.fov = (FOVx, FOVx)

        # Slice thickness and spacing
        if slice_thickness is None:
            raise ValueError("Please provide slice_thickness")
        self.slice_thickness = slice_thickness
        self.slice_spacing = slice_spacing

        # Build matrix
        if Nx is None:
            raise ValueError("Please provide Nx")
        if Ny is None:
            self.matrix_size = (Nx, Ny)
        else:
            self.matrix_size = (Nx, Nx)

        # Number of slices
        if Nslices is None:
            raise ValueError("Please provide Nslices")
        self.n_slices = Nslices

        # Flip angle
        if flip is None:
            raise ValueError("Please provide flip")
        self.flip_angle = flip

        # TE / TR
        self.TE = TE
        self.TR = TR

        # Accelerations
        self.R = Rplane
        self.PF = PFplane

        # Build opts
        super().__init__(
            dwell,
            raster,
            gmax,
            smax,
            b0_field,
            rf_dead_time,
            rf_ringdown_time,
            adc_dead_time,
            psd_rf_wait,
        )


class Cartesian3DParams(BaseParams):
    """
    Parse parameters for single-echo 3D Cartesian imaging.

    This is designed to convert data parsed by ParamsParser
    to a format suitable for single-echo 3D Cartesian design routines.

    """

    def __init__(
        self,
        FOVx: float | None = None,
        FOVy: float | None = None,
        Nx: int | None = None,
        Ny: int | None = None,
        Nslices: int | None = None,
        slice_thickness: float | None = None,
        Rplane: float | None = 1,
        Rplane2: float | None = 1,
        PFslice: float | None = 1.0,
        TE: float | None = 0.0,
        TR: float | None = 0.0,
        flip: float | None = None,
        dwell: float | None = 4e-6,
        raster: float | None = 2e-6,
        gmax: float | None = None,
        smax: float | None = None,
        b0_field: float | None = None,
        rf_dead_time: float | None = 100e-6,
        rf_ringdown_time: float | None = 60e-6,
        adc_dead_time: float | None = 40e-6,
        psd_rf_wait: float | None = 0.0,
        *args,
        **kwargs,
    ):  # noqa
        # Slice thickness and spacing
        if slice_thickness is None:
            raise ValueError("Please provide slice_thickness")

        # Number of slices
        if Nslices is None:
            raise ValueError("Please provide Nslices")

        # Build FOV
        if FOVx is None:
            raise ValueError("Please provide FOVx")
        if FOVy is None:
            self.fov = (FOVx, FOVy, slice_thickness * Nslices)
        else:
            self.fov = (FOVx, FOVx, slice_thickness * Nslices)

        # Build matrix
        if Nx is None:
            raise ValueError("Please provide Nx")
        if Ny is None:
            self.matrix_size = (Nx, Ny, Nslices)
        else:
            self.matrix_size = (Nx, Nx, Nslices)

        # Flip angle
        if flip is None:
            raise ValueError("Please provide flip")
        self.flip_angle = flip

        # TE / TR
        self.TE = TE
        self.TR = TR

        # Accelerations
        self.Rpi = Rplane
        self.Rcs = Rplane2
        self.PF = PFslice

        # Build opts
        super().__init__(
            dwell,
            raster,
            gmax,
            smax,
            b0_field,
            rf_dead_time,
            rf_ringdown_time,
            adc_dead_time,
            psd_rf_wait,
        )
