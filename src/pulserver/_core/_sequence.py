"""Intermediate sequence representation."""

__all__ = ["Sequence"]

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from ._ceq import Ceq, PulseqBlock


class Sequence:
    """
    Pulseq Sequence intermediate representation.

    This is related to Pulceq PulseqBlock structure.

    Each block is a collection of one or more PyPulseq events. For each PulseqBlock,
    a maximum of 1 event for each board (rf, gx, gy, gz, adc, trig) can be executed.

    """

    def __init__(self, system: SimpleNamespace, format: str | bool):
        self._system = system
        self._format = format

        if self._format == "pulseq":
            self._sequence = pp.Sequence(system=system)
        else:
            self._loop = []

        if self._format == "pulseq":
            self._block_library = {"delay": {}}
        else:
            self._block_library = {"delay": PulseqBlock(ID=0)}

        self._section_labels = []
        self._sections_edges = []

    def register_block(
        self,
        name: str,
        rf: SimpleNamespace | None = None,
        gx: SimpleNamespace | None = None,
        gy: SimpleNamespace | None = None,
        gz: SimpleNamespace | None = None,
        adc: SimpleNamespace | None = None,
        trig: SimpleNamespace | None = None,
    ):
        # sanity checks
        if self._format == "pulseq":
            assert (
                len(self._sequence.block_events) == 0
            ), "Please define all the events before building the loop."
        else:
            assert (
                len(self._loop) == 0
            ), "Please define all the events before building the loop."
        if rf is not None and adc is not None:
            VALID_BLOCK = False
        else:
            VALID_BLOCK = True
        assert VALID_BLOCK, "Error! A block cannot contain both a RF and ADC event."
        if gx is not None:
            assert (
                gx.channel == "x"
            ), f"x-gradient waveform is directed towards {gx.channel}"
        if gy is not None:
            assert (
                gy.channel == "y"
            ), f"y-gradient waveform is directed towards {gy.channel}"
        if gz is not None:
            assert (
                gz.channel == "z"
            ), f"z-gradient waveform is directed towards {gz.channel}"

        # update block library
        if self._format == "pulseq":
            self._block_library[name] = {}
            if rf is not None:
                self._block_library[name]["rf"] = deepcopy(rf)
            if gx is not None:
                self._block_library[name]["gx"] = deepcopy(gx)
            if gy is not None:
                self._block_library[name]["gy"] = deepcopy(gy)
            if gz is not None:
                self._block_library[name]["gz"] = deepcopy(gz)
            if adc is not None:
                self._block_library[name]["adc"] = deepcopy(adc)
            if trig is not None:
                self._block_library[name]["trig"] = deepcopy(trig)
        else:
            ID = len(self._block_library)
            self._block_library[name] = PulseqBlock(ID, rf, gx, gy, gz, adc, trig)

    def section(self, name: str):
        assert (
            name not in self._section_labels
        ), f"Section {name} already exists - please use another name."
        if self._format == "pulseq":
            _current_seqlength = len(self._sequence.block_events)
        else:
            _current_seqlength = len(self._loop)
        self._sections_edges.append(_current_seqlength)

    def add_block(
        self,
        name: str,
        gx_amp: float = 1.0,
        gy_amp: float = 1.0,
        gz_amp: float = 1.0,
        rf_amp: float = 1.0,
        rf_phase: float = 0.0,
        rf_freq: float = 0.0,
        adc_phase: float = 0.0,
        delay: float | None = None,
        rotmat: np.ndarray | None = None,
    ):
        assert name in self._block_library, f"Requested block {name} not found!"
        if self._format == "pulseq":
            current_block = deepcopy(self._block_library[name])

            # scale RF pulse and apply phase modulation / frequency offset
            if "rf" in current_block:
                current_block["rf"].signal *= rf_amp
                current_block["rf"].phase_offset = rf_phase
                current_block["rf"].freq_offset += rf_freq

            # apply phase modulation to ADC
            if "adc" in current_block:
                current_block["adc"].phase_offset = adc_phase

            # scale gradients
            if "gx" in current_block:
                current_block["gx"] = pp.scale_grad(
                    grad=current_block["gx"], scale=gx_amp
                )
            if "gy" in current_block:
                current_block["gy"] = pp.scale_grad(
                    grad=current_block["gy"], scale=gy_amp
                )
            if "gz" in current_block:
                current_block["gz"] = pp.scale_grad(
                    grad=current_block["gz"], scale=gy_amp
                )

            # rotate gradients
            if rotmat is not None:
                # extract gradient waveforms from current event
                current_grad = {}
                for ch in ["gx", "gy", "gz"]:
                    if ch in current_block:
                        current_grad[ch] = current_block[ch]

                # actual rotation
                current_block = _pp_rotate(current_block, rotmat)

                # replace rotated gradients in current event
                for ch in ["gx", "gy", "gz"]:
                    if ch in current_block:
                        current_block[ch] = current_grad[ch]

            # update sequence structure
            if delay is not None:
                self._sequence.add_block(pp.make_delay(delay), *current_block.values())
            else:
                self._sequence.add_block(*current_block.values())

        else:
            parent_block_id = self._block_library[name].ID
            block_duration = self._block_library[name].block_duration
            if delay is not None:
                block_duration = max(delay, block_duration)
            if rotmat is None:
                rotmat = np.eye(3, dtype=np.float32).ravel().tolist()
                hasrot = [1]
            else:
                rotmat = rotmat.ravel().tolist()
                hasrot = [-1]

            loop_row = (
                [
                    -1,
                    parent_block_id,
                    rf_amp,
                    rf_phase,
                    rf_freq,
                    gx_amp,
                    gy_amp,
                    gz_amp,
                    adc_phase,
                    block_duration,
                ]
                + rotmat
                + hasrot
            )
            self._loop.append(loop_row)

    def export(self):
        if self._format == "pulseq":
            return self._sequence

        # prepare Ceq structure
        self._sequence = Ceq(
            list(self._block_library.values()),
            self._loop,
            self._sections_edges,
        )

        # if required, dump to bytes string
        if self._format == "bytes":
            self._sequence = self._sequence.to_bytes()

        return self._sequence


def _pp_rotate(grad, rot_matrix):
    grad_channels = ["gx", "gy", "gz"]
    grad = deepcopy(grad)

    # get length of gradient waveforms
    wave_length = []
    for ch in grad_channels:
        if ch in grad:
            wave_length.append(len(grad[ch]))

    assert (
        np.unique(wave_length) != 0
    ).sum() == 1, "All the waveform along different channels must have the same length"

    wave_length = np.unique(wave_length)
    wave_length = wave_length[wave_length != 0].item()

    # create zero-filled waveforms for empty gradient channels
    for ch in grad_channels:
        if ch in grad:
            grad[ch] = grad[ch].squeeze()
        else:
            grad[ch] = np.zeros(wave_length)

    # stack matrix
    grad_mat = np.stack(
        (grad["gx"], grad["gy"], grad["gz"]), axis=0
    )  # (3, wave_length)

    # apply rotation
    grad_mat = rot_matrix @ grad_mat

    # put back in dictionary
    for j in range(3):
        ch = grad_channels[j]
        grad[ch] = grad_mat[j]

    # remove all zero waveforms
    for ch in grad_channels:
        if np.allclose(grad[ch], 0.0):
            grad.pop(ch)

    return grad
