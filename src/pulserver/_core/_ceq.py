"""Ceq structure definition."""

__all__ = ["Ceq", "PulseqBlock", "SequenceParams"]

from dataclasses import dataclass
from dataclasses import asdict as _asdict
from types import SimpleNamespace
from typing import Optional

import struct
import numpy as np
import pypulseq as pp

from . import _autoseg

CHANNEL_ENUM = {"osc0": 0, "osc1": 1, "ext1": 2}
SEGMENT_RINGDOWN_TIME = 116 * 1e-6  # TODO: doesn't have to be hardcoded


@dataclass
class PulseqShapeArbitrary:
    n_samples: int
    raster: float
    time: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray | None = None
    amplitude: float | None = None

    def __post_init__(self):
        self.magnitude = np.asarray(self.magnitude, dtype=np.float32)
        self.time = (
            np.asarray(self.time, dtype=np.float32) if self.time is not None else None
        )
        self.phase = (
            np.asarray(self.phase, dtype=np.float32) if self.phase is not None else None
        )

        # determine amplitude and normalize waveform
        self.amplitude = abs(self.magnitude).max()
        self.magnitude *= 32767 / self.amplitude

    def to_bytes(self, endian=">") -> bytes:
        _bytes = struct.pack(endian + "i", self.n_samples) + struct.pack(
            endian + "f", self.raster
        )

        # add time
        if self.time is not None:
            _bytes += self.time.astype(endian + "f4").tobytes()

        # add magnitude
        _bytes += self.magnitude.astype(endian + "f4").tobytes()

        # add phase
        if self.phase is not None:
            _bytes += self.phase.astype(endian + "f4").tobytes()

        # add amplitude
        _bytes += struct.pack(endian + "f", self.amplitude)

        return _bytes


@dataclass
class PulseqShapeTrap:
    amplitude: float
    rise_time: float
    flat_time: float
    fall_time: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "f", self.amplitude)
            + struct.pack(endian + "f", self.rise_time)
            + struct.pack(endian + "f", self.flat_time)
            + struct.pack(endian + "f", self.fall_time)
        )


@dataclass
class PulseqRF:
    type: int
    wav: PulseqShapeArbitrary
    duration: float
    delay: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + self.wav.to_bytes(endian)
            + struct.pack(endian + "f", self.duration)
            + struct.pack(endian + "f", self.delay)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqRF":
        n_samples = data.signal.shape[0]

        # determine whether wave is arbitrary or extended trap
        dt = np.unique(np.diff(np.round(data.t * 1e6))) / 1e6

        if len(dt) == 1:  # uniform raster -> arbitrary shape
            type = 1
            raster = dt.item()
            time = None
        else:  # non-uniform raster -> extended trapezoid
            type = 2
            raster = 0.0
            time = data.t

        rho = np.abs(data.signal)
        theta = np.angle(data.signal)

        return cls(
            type=type,
            wav=PulseqShapeArbitrary(n_samples, raster, time, rho, theta),
            duration=data.shape_dur,
            delay=data.delay,
        )


@dataclass
class PulseqGrad:
    type: int
    delay: float
    shape: PulseqShapeArbitrary | PulseqShapeTrap

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "f", self.delay)
            + self.shape.to_bytes(endian)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqGrad":
        print(data)
        if data.type == "trap":
            type = 1
            shape_obj = PulseqShapeTrap(
                data.amplitude, data.rise_time, data.flat_time, data.fall_time
            )
        elif data.type == "grad":
            n_samples = data.waveform.shape[0]

            # determine whether wave is arbitrary or extended trap
            dt = np.unique(np.diff(np.round(data.tt * 1e6))) / 1e6

            if len(dt) == 1:  # uniform raster -> arbitrary shape
                print("arbitrary")
                type = 2
                raster = dt.item()
                time = None
            else:  # non-uniform raster -> extended trapezoid
                print("extended")
                type = 3
                raster = 0.0
                time = data.tt

            waveform = data.waveform
            shape_obj = PulseqShapeArbitrary(n_samples, raster, time, waveform)

        return cls(type=type, delay=data.delay, shape=shape_obj)


@dataclass
class PulseqADC:
    type: int
    num_samples: int
    dwell: float
    delay: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "i", self.num_samples)
            + struct.pack(endian + "f", self.dwell)
            + struct.pack(endian + "f", self.delay)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqADC":
        return cls(
            type=1,
            num_samples=data.num_samples,
            dwell=data.dwell,
            delay=data.delay,
        )


@dataclass
class PulseqTrig:
    type: int
    channel: int
    delay: float
    duration: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "i", self.channel)
            + struct.pack(endian + "f", self.delay)
            + struct.pack(endian + "f", self.duration)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqTrig":
        return cls(
            type=1,
            channel=CHANNEL_ENUM[data.channel],
            delay=data.delay,
            duration=data.duration,
        )


class PulseqBlock:
    """Pulseq block structure."""

    def __init__(
        self,
        ID: int,
        rf: SimpleNamespace = None,
        gx: SimpleNamespace = None,
        gy: SimpleNamespace = None,
        gz: SimpleNamespace = None,
        adc: SimpleNamespace = None,
        trig: SimpleNamespace = None,
        delay: SimpleNamespace = None,
    ) -> "PulseqBlock":
        self.ID = ID
        args = [rf, gx, gy, gz, adc, trig, delay]
        args = [arg for arg in args if arg is not None]
        self.duration = pp.calc_duration(*args)
        self.rf = PulseqRF.from_struct(rf) if rf else None
        self.gx = PulseqGrad.from_struct(gx) if gx else None
        self.gy = PulseqGrad.from_struct(gy) if gy else None
        self.gz = PulseqGrad.from_struct(gz) if gz else None
        self.adc = PulseqADC.from_struct(adc) if adc else None
        self.trig = PulseqTrig.from_struct(trig) if trig else None

    def to_bytes(self, endian=">") -> bytes:  # noqa
        bytes_data = struct.pack(endian + "i", self.ID) + struct.pack(
            endian + "f", self.duration
        )

        # RF Event
        if self.rf:
            bytes_data += self.rf.to_bytes(endian)
        else:
            bytes_data += struct.pack(endian + "h", 0)

        # Gradient Events
        for grad in [self.gx, self.gy, self.gz]:
            if grad:
                bytes_data += grad.to_bytes(endian)
            else:
                bytes_data += struct.pack(endian + "h", 0)

        # ADC Event
        if self.adc:
            bytes_data += self.adc.to_bytes(endian)
        else:
            bytes_data += struct.pack(endian + "h", 0)

        # Trigger Event
        if self.trig:
            bytes_data += self.trig.to_bytes(endian)
        else:
            bytes_data += struct.pack(endian + "h", 0)

        return bytes_data


class Segment:
    """Ceq segment."""

    def __init__(self, segment_id: int, block_ids: list[int]):
        self.segment_id = segment_id
        self.n_blocks_in_segment = len(block_ids)
        self.block_ids = np.asarray(block_ids, dtype=np.int16)

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.segment_id)
            + struct.pack(endian + "h", self.n_blocks_in_segment)
            + self.block_ids.astype(endian + "i2").tobytes()
        )


class Ceq:
    """CEQ structure."""

    def __init__(
        self,
        parent_blocks: list[PulseqBlock],
        loop: list[list],
        sections_edges: list[list[int]],
    ):
        loop = np.asarray(loop, dtype=np.float32)
        segments = _build_segments(loop, sections_edges)

        # build CEQ structure
        self.n_max = loop.shape[0]
        self.n_parent_blocks = len(parent_blocks)
        self.n_segments = len(segments)
        self.parent_blocks = parent_blocks
        self.segments = segments
        self.n_columns_in_loop_array = loop.shape[1] - 2  # discard "hasrot", "hasadc"
        self.loop = loop[:, :-2]
        self.max_b1 = _find_b1_max(parent_blocks)
        self.duration = _calc_duration(self.loop[:, 0], self.loop[:, 9])
        self.n_readouts = int(np.sum(loop[:, -2]))

    def to_bytes(self, endian=">") -> bytes:  # noqa
        bytes_data = (
            struct.pack(endian + "i", self.n_max)
            + struct.pack(endian + "h", self.n_parent_blocks)
            + struct.pack(endian + "h", self.n_segments)
        )
        for block in self.parent_blocks:
            bytes_data += block.to_bytes(endian)
        for segment in self.segments:
            bytes_data += segment.to_bytes(endian)
        bytes_data += struct.pack(endian + "h", self.n_columns_in_loop_array)
        bytes_data += self.loop.astype(endian + "f4").tobytes()
        bytes_data += struct.pack(endian + "f", self.max_b1)
        bytes_data += struct.pack(endian + "f", self.duration)
        bytes_data += struct.pack(endian + "i", self.n_readouts)

        return bytes_data

    def export(self, dformat="file"):  # noqa
        if dformat == "bytes":
            return self.to_bytes(endian=">")
        elif dformat == "file":
            return self.to_bytes(endian="<")


@dataclass
class SequenceParams:
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

    @classmethod
    def from_bytes(cls, data: bytes) -> "SequenceParams":
        """
        Deserialize from a byte array into a SequenceParams object.

        Parameters
        ----------
        data : bytes
            A byte array representing a serialized SequenceParams object.

        Returns
        -------
        SequenceParams
            A SequenceParams instance with attributes filled.
        """
        format_string = "2f 5h 2f 5f h 5f 3f 3h h 6f"
        unpacked = struct.unpack(format_string, data)

        # Replace -1 values with None
        unpacked = [None if x == -1 or x == -1.0 else x for x in unpacked]

        return cls(*unpacked)

    def asdict(self) -> dict:
        """
        Return a dictionary of the dataclass, excluding None values.

        Returns
        -------
        dict
            A dictionary of the dataclass fields, excluding None values.
        """
        return {k: v for k, v in _asdict(self).items() if v is not None}


# %% local subroutines
def _build_segments(loop, sections_edges):
    hasrot = np.ascontiguousarray(loop[:, -1]).astype(int)
    parent_block_id = np.ascontiguousarray(loop[:, 1]).astype(int) * hasrot

    # build section edges
    if not sections_edges:
        sections_edges = [0]
    sections_edges = np.stack((sections_edges, sections_edges[1:] + [-1]), axis=-1)

    # loop over sections and find segment definitions
    segment_id = np.zeros(loop.shape[0], dtype=np.float32)
    seg_definitions = []

    # fill sections from 0 to n-1
    n_sections = len(sections_edges)
    for n in range(n_sections - 1):
        section_start, section_end = sections_edges[n]
        _seg_definition = _autoseg.find_segment_definitions(
            parent_block_id[section_start:section_end]
        )
        _seg_definition = _autoseg.split_rotated_segments(_seg_definition)
        seg_definitions.extend(_seg_definition)

    # fill last section
    section_start = sections_edges[-1][0]
    _seg_definition = _autoseg.find_segment_definitions(parent_block_id[section_start:])
    _seg_definition = _autoseg.split_rotated_segments(_seg_definition)
    seg_definitions.extend(_seg_definition)

    # for each block, find the segment it belongs to
    for n in range(len(seg_definitions)):
        idx = _autoseg.find_segments(parent_block_id, seg_definitions[n])
        segment_id[idx] = n
    loop[:, 0] = segment_id

    # now build segment fields
    n_segments = len(seg_definitions)
    segments = []
    for n in range(n_segments):
        segments.append(Segment(n, seg_definitions[n]))

    return segments


def _find_b1_max(parent_blocks):
    return np.max(
        [block.rf.wav.amplitude for block in parent_blocks if block.rf is not None]
    )


def _calc_duration(segment_id, block_duration):
    block_duration = block_duration.sum()

    # total segment ringdown
    n_seg_boundaries = (np.diff(segment_id) != 0).sum()
    seg_ringdown_duration = SEGMENT_RINGDOWN_TIME * n_seg_boundaries

    return block_duration + seg_ringdown_duration
