"""Ceq structure definition."""

__all__ = ["Ceq", "PulseqBlock"]

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Union

import struct
import numpy as np
import pypulseq as pp

from . import _autoseg

CHANNEL_ENUM = {"osc0": 0, "osc1": 1, "ext1": 2}
SEGMENT_RINGDOWN_TIME = 116 * 1e-6


@dataclass
class PulseqShapeArbitrary:
    n_samples: int
    raster: float
    waveform: np.ndarray
    # phase: np.ndarray = None # TODO: consider removing

    def __post_init__(self):
        self.waveform = np.asarray(self.waveform, dtype=np.float32)
        # self.phase = np.asarray(self.phase, dtype=np.float32) if self.phase else 0 * self.magnitude # TODO: consider removing

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">i", self.n_samples)
            + struct.pack(">f", self.raster)
            + self.waveform.astype(">f4").tobytes()  # +
            # self.phase.astype(">f4").tobytes()
        )


@dataclass
class PulseqShapeTrap:
    rise_time: float
    flat_time: float
    fall_time: float

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">f", self.rise_time)
            + struct.pack(">f", self.flat_time)
            + struct.pack(">f", self.fall_time)
        )


@dataclass
class PulseqRF:
    type: int
    n_samples: int
    rho: np.ndarray
    theta: np.ndarray
    t: np.ndarray
    shape_dur: float
    delay: float
    freq_offset: float
    phase_offset: float
    max_b1: float

    def __post_init__(self):
        self.rho = np.asarray(self.rho, dtype=np.float32)
        self.theta = np.asarray(self.theta, dtype=np.float32)
        self.t = np.asarray(self.t, dtype=np.float32)

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">h", self.type)
            + struct.pack(">h", self.n_samples)
            + self.rho.astype(">f4").tobytes()
            + self.theta.astype(">f4").tobytes()
            + self.t.astype(">f4").tobytes()
            + struct.pack(">f", self.shape_dur)
            + struct.pack(">f", self.delay)
            + struct.pack(">f", self.freq_offset)
            + struct.pack(">f", self.phase_offset)
            + struct.pack(">f", self.max_b1)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqRF":
        return cls(
            type=1,
            n_samples=data.signal.shape[0],
            rho=np.abs(data.signal),
            theta=np.angle(data.signal),
            t=data.t,
            shape_dur=data.shape_dur,
            delay=data.delay,
            freq_offset=data.freq_offset,
            phase_offset=data.phase_offset,
            max_b1=max(abs(data.signal)),
        )


@dataclass
class PulseqGrad:
    type: int
    amplitude: float
    delay: float
    shape: Union[PulseqShapeArbitrary, PulseqShapeTrap]

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">h", self.type)
            + struct.pack(">f", self.amplitude)
            + struct.pack(">f", self.delay)
            + self.shape.to_bytes()
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqGrad":
        if data.type == "trap":
            type = 1
            amplitude = data.amplitude
            shape_obj = PulseqShapeTrap(data.rise_time, data.flat_time, data.fall_time)
        elif data.type == "grad":
            type = 2
            amplitude = max(abs(data.waveform))
            waveform = data.waveform / amplitude if amplitude != 0.0 else data.waveform
            n_samples = data.waveform.shape[0]
            raster = np.diff(data.tt)[0]
            shape_obj = PulseqShapeArbitrary(n_samples, raster, waveform)
        return cls(type=type, amplitude=amplitude, delay=data.delay, shape=shape_obj)


@dataclass
class PulseqADC:
    type: int
    num_samples: int
    dwell: float
    delay: float
    freq_offset: float
    phase_offset: float

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">h", self.type)
            + struct.pack(">i", self.num_samples)
            + struct.pack(">f", self.dwell)
            + struct.pack(">f", self.delay)
            + struct.pack(">f", self.freq_offset)
            + struct.pack(">f", self.phase_offset)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqADC":
        return cls(
            type=1,
            num_samples=data.num_samples,
            dwell=data.dwell,
            delay=data.delay,
            freq_offset=data.freq_offset,
            phase_offset=data.phase_offset,
        )


@dataclass
class PulseqTrig:
    type: int
    channel: int
    delay: float
    duration: float

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">i", self.type)
            + struct.pack(">i", self.channel)
            + struct.pack(">f", self.delay)
            + struct.pack(">f", self.duration)
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
    ) -> "PulseqBlock":
        self.ID = ID
        args = [rf, gx, gy, gz, adc, trig]
        args = [arg for arg in args if arg is not None]
        self.block_duration = pp.calc_duration(*args)
        self.rf = PulseqRF.from_struct(rf) if rf else None
        self.gx = PulseqGrad.from_struct(gx) if gx else None
        self.gy = PulseqGrad.from_struct(gy) if gy else None
        self.gz = PulseqGrad.from_struct(gz) if gz else None
        self.adc = PulseqADC.from_struct(adc) if adc else None
        self.trig = PulseqTrig.from_struct(trig) if trig else None

    def to_bytes(self) -> bytes:
        bytes_data = struct.pack(">i", self.ID) + struct.pack(">f", self.block_duration)

        # RF Event
        if self.rf:
            bytes_data += self.rf.to_bytes()
        else:
            bytes_data += struct.pack(">h", 0)  # * 2

        # Gradient Events
        for grad in [self.gx, self.gy, self.gz]:
            if grad:
                bytes_data += grad.to_bytes()
            else:
                # * 2 + struct.pack(">f", 0) * 2
                bytes_data += struct.pack(">h", 0)

        # ADC Event
        if self.adc:
            bytes_data += self.adc.to_bytes()
        else:
            bytes_data += struct.pack(">h", 0)  # * 6

        # Trigger Event
        if self.trig:
            bytes_data += self.trig.to_bytes()
        else:
            # * 2 + struct.pack(">f", 0) * 2
            bytes_data += struct.pack(">i", 0)

        return bytes_data


class Segment:
    """Ceq segment."""

    def __init__(self, segment_id: int, block_ids: list[int]):
        self.segment_id = segment_id
        self.n_blocks_in_segment = len(block_ids)
        self.block_ids = np.asarray(block_ids, dtype=np.int16)

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">h", self.segment_id)
            + struct.pack(">h", self.n_blocks_in_segment)
            + self.block_ids.astype(">i2").tobytes()
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
        self.segments = segments
        self.n_columns_in_loop_array = loop.shape[1] - 1  # discard "hasrot"
        self.loop = loop[:, :-1]
        self.max_b1 = _find_b1_max(parent_blocks)
        self.duration = _calc_duration(self.loop[:, 0], self.loop[:, 9])

    def to_bytes(self) -> bytes:
        bytes_data = (
            struct.pack(">i", self.n_max)
            + struct.pack(">h", self.n_parent_blocks)
            + struct.pack(">h", self.n_segments)
        )
        for block in self.parent_blocks:
            bytes_data += block.to_bytes()
        for segment in self.segments:
            bytes_data += segment.to_bytes()
        bytes_data += struct.pack(">h", self.n_columns_in_loop_array)
        bytes_data += self.loop.astype(">f4").tobytes()
        bytes_data += struct.pack(">f", self.max_b1)
        bytes_data += struct.pack(">f", self.duration)
        return bytes_data


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
    for n in range(n_sections-1):
        section_start, section_end = sections_edges[n]
        _seg_definition = _autoseg.find_segment_definitions(
            parent_block_id[section_start:section_end]
        )
        _seg_definition = _autoseg.split_rotated_segments(_seg_definition)
        seg_definitions.extend(_seg_definition)
        
    # fill last section
    section_start = sections_edges[-1][0]
    _seg_definition = _autoseg.find_segment_definitions(
        parent_block_id[section_start:]
    )
    _seg_definition = _autoseg.split_rotated_segments(_seg_definition)
    seg_definitions.extend(_seg_definition)

    # for each event, find the segment it belongs to
    for n in range(len(seg_definitions)):
        idx = _autoseg.find_segments(parent_block_id, seg_definitions[n])
        segment_id[idx] = n
    segment_id += 1  # segment 0 is reserved for pure delay
    loop[:, 0] = segment_id

    # now build segment fields
    n_segments = len(seg_definitions)
    segments = []
    for n in range(n_segments):
        segments.append(Segment(n + 1, seg_definitions[n]))

    return segments


def _find_b1_max(parent_blocks):
    return np.max([block.rf.max_b1 for block in parent_blocks if block.rf is not None])


def _calc_duration(segment_id, block_duration):
    block_duration = block_duration.sum()

    # total segment ringdown
    n_seg_boundaries = (np.diff(segment_id) != 0).sum()
    seg_ringdown_duration = SEGMENT_RINGDOWN_TIME * n_seg_boundaries

    return block_duration + seg_ringdown_duration
