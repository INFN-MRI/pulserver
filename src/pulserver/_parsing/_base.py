"""Sequence Parameters object."""

__all__ = ["ParamsParser"]

from dataclasses import dataclass, fields
from dataclasses import asdict as _asdict

import struct


@dataclass
class ParamsParser:
    """
    Python representation of the C SequenceParams struct.

    Attributes
    ----------
    FOVx : Optional[float]
        Field of view in mm (x-direction).
    FOVy : Optional[float]
        Field of view in mm (y-direction).
    Nx : Optional[int]
        Matrix size (x-direction).
    Ny : Optional[int]
        Matrix size (y-direction).
    Nslices : Optional[int]
        Number of slices.
    Nechoes : Optional[int]
        Number of echoes.
    Nphases : Optional[int]
        Number of phases.
    slice_thickness : Optional[float]
        Thickness of each slice (mm).
    slice_spacing : Optional[float]
        Spacing between slices (mm).
    Rplane : Optional[float]
        In-plane undersampling factor.
    Rplane2 : Optional[float]
        Additional in-plane undersampling factor.
    Rslice : Optional[float]
        Through-plane undersampling factor.
    PFplane : Optional[float]
        In-plane partial fourier.
    PFslice : Optional[float]
        Through-plane partial fourier.
    ETL : Optional[int]
        Number of k-space shots per readout.
    TE : Optional[float]
        Echo time (ms).
    TE0 : Optional[float]
        First echo time (ms) for multiecho.
    TR : Optional[float]
        Repetition time (ms).
    Tprep : Optional[float]
        Preparation time (ms).
    Trecovery : Optional[float]
        Recovery time (ms).
    flip : Optional[float]
        Flip angle in degrees.
    flip2 : Optional[float]
        Second flip angle in degrees.
    refoc_flip : Optional[float]
        Refocusing flip angle in degrees.
    freq_dir : Optional[int]
        Frequency direction (0: A/P; 1: S/L).
    freq_verse : Optional[int]
        Frequency verse (1: normal; -1: swapped).
    phase_verse : Optional[int]
        Phase verse (1: normal; -1: swapped).
    bipolar_echoes : Optional[int]
        Bipolar echoes (0: false, 1: true).
    dwell : Optional[float]
        ADC dwell time (s).
    raster : Optional[float]
        Waveform raster time (s).
    gmax : Optional[float]
        Maximum gradient strength (mT/m).
    smax : Optional[float]
        Maximum gradient slew rate (T/m/s).
    b1_max : Optional[float]
        Maximum RF value (uT).
    b0_field : Optional[float]
        System field strength (T).
    """

    function_name: str
    FOVx: float | None = None
    FOVy: float | None = None
    Nx: int | None = None
    Ny: int | None = None
    Nslices: int | None = None
    Nechoes: int | None = None
    Nphases: int | None = None
    slice_thickness: float | None = None
    slice_spacing: float | None = None
    Rplane: float | None = None
    Rplane2: float | None = None
    Rslice: float | None = None
    PFplane: float | None = None
    PFslice: float | None = None
    ETL: int | None = None
    TE: float | None = None
    TE0: float | None = None
    TR: float | None = None
    Tprep: float | None = None
    Trecovery: float | None = None
    flip: float | None = None
    flip2: float | None = None
    refoc_flip: float | None = None
    freq_dir: int | None = None
    freq_verse: int | None = None
    phase_verse: int | None = None
    bipolar_echoes: int | None = None
    dwell: float | None = None
    raster: float | None = None
    gmax: float | None = None
    smax: float | None = None
    b1_max: float | None = None
    b0_field: float | None = None
    rf_dead_time: float | None = None
    rf_ringdown_time: float | None = None
    adc_dead_time: float | None = None
    psd_rf_wait: float | None = None
    psd_grd_wait: float | None = None

    @classmethod
    def from_bytes(cls, data: bytes) -> "ParamsParser":
        """Deserialize from a byte array into a SequenceParams object."""
        format_string = "2f 5h 7f h 8f 4h 11f"

        # Unpack the function name
        function_name = struct.unpack("50s", data[:50])[0]
        function_name = function_name.decode("utf-8").rstrip("\x00")

        # Unpack values
        values = struct.unpack(format_string, data[50:])
        values = [None if x == -1 or x == -1.0 else x for x in values]

        return ParamsParser(function_name, *values)

    def to_bytes(self) -> bytes:  # noqa
        """
        Serialize this dataclass to a byte array.
        """
        format_string = "2f 5h 7f h 8f 4h 11f"
        field_types = [field.type for field in fields(self.__class__)][1:]

        # Pack function name
        function_name = struct.pack("50s", self.function_name.encode("utf-8"))

        # Pack values
        values = list(self.asdict(filt=False).values())
        values = [-1 if x is None else x for x in values]
        values = [field_types[n].__args__[0](values[n]) for n in range(len(values))]
        values = struct.pack(format_string, *values)

        return function_name + values

    def asdict(self, filt=True) -> dict:
        """
        Return a dictionary of the dataclass, excluding None values.

        Returns
        -------
        dict
            A dictionary of the dataclass fields, excluding None values.
        """
        if filt:
            out = {k: v for k, v in _asdict(self).items() if v is not None}
        else:
            out = _asdict(self)

        out.pop("function_name")
        return out
