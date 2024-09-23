"""Ceq structure definition."""

__all__ = ["Ceq"]

from dataclasses import dataclass
from types import SimpleNamespace

from typing import Dict
from typing import Optional
from typing import Union

import struct
import numpy as np

CHANNEL_ENUM = {"osc0": 0, "osc1": 1, "ext1": 2}


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


@dataclass
class PulseqBlock:
    ID: int
    block_duration: float
    rf: Optional[PulseqRF]
    gx: Optional[PulseqGrad]
    gy: Optional[PulseqGrad]
    gz: Optional[PulseqGrad]
    adc: Optional[PulseqADC]
    trig: Optional[PulseqTrig]

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
                bytes_data += struct.pack(">h", 0)  # * 2 + struct.pack(">f", 0) * 2

        # ADC Event
        if self.adc:
            bytes_data += self.adc.to_bytes()
        else:
            bytes_data += struct.pack(">h", 0)  # * 6

        # Trigger Event
        if self.trig:
            bytes_data += self.trig.to_bytes()
        else:
            bytes_data += struct.pack(">i", 0)  # * 2 + struct.pack(">f", 0) * 2

        return bytes_data

    @classmethod
    def from_dict(cls, data: Dict) -> "PulseqBlock":
        return cls(
            ID=data["ID"],
            block_duration=data["block_duration"],
            rf=PulseqRF.from_struct(data["rf"]) if data.get("rf") else None,
            gx=PulseqGrad.from_struct(data["gx"]) if data.get("gx") else None,
            gy=PulseqGrad.from_struct(data["gy"]) if data.get("gy") else None,
            gz=PulseqGrad.from_struct(data["gz"]) if data.get("gz") else None,
            adc=PulseqADC.from_struct(data["adc"]) if data.get("adc") else None,
            trig=PulseqTrig.from_struct(data["trig"]) if data.get("trig") else None,
        )


@dataclass
class Segment:
    segment_id: int
    n_blocks_in_segment: int
    block_ids: np.ndarray

    def __post_init__(self):
        self.block_ids = np.asarray(self.block_ids, dtype=np.int16)

    def to_bytes(self) -> bytes:
        return (
            struct.pack(">h", self.segment_id)
            + struct.pack(">h", self.n_blocks_in_segment)
            + self.block_ids.astype(">i2").tobytes()
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "Segment":
        return cls(
            segment_id=data["segment_id"],
            n_blocks_in_segment=data["n_blocks_in_segment"],
            block_ids=data["block_ids"],
        )


@dataclass
class Ceq:
    n_max: int = 0  # number of blocks (rows in BLOCKS section) in .seq file
    n_parent_blocks: int = 0
    n_segments: int = 1
    parent_blocks: list[PulseqBlock] = None
    segments: list[Segment] = None
    n_columns_in_loop_array: int = 10  # TODO: add rotmat (+= 9 elements)
    loop: list[
        np.ndarray
    ] = None  # dynamic scan settings of shape (n_max, n_columns_in_loop_array), loop[n] = [segment_id ID rfamp rfphs rffreq gxamp gyamp gzamp recphs block_duration]
    max_b1: float = 0.0  # Hz. Max B1 amplitude across all parent blocks
    duration: float = 0.0  # scan duration, sec

    def __post_init__(self):
        self.parent_blocks = []
        self.segments = []
        self.loop = []

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
