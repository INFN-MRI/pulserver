"""Intermediate sequence representation."""

__all__ = ["Sequence"]

from copy import copy
from typing import SimpleNamespace

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

        self._event_id = {"delay": 0}

        if self._format == "pulseq":
            self._event_library = {"delay": {}}
        else:
            self._event_library = {"delay": PulseqBlock(ID=0)}

        self._sections_edges = []


    def register_event(
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
                self._sequence.n_max == 0
            ), "Please define all the events before building the loop."
        if rf is not None and adc is not None:
            VALID_BLOCK = False
        else:
            VALID_BLOCK = True
        assert VALID_BLOCK, "Error! A block cannot contain both a RF and ADC event."

        # update event library
        if self._format == "pulseq":
            self._event_library[name] = {
                "rf": rf,
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "adc": adc,
                "trig": trig,
            }
        else:
            ID = len(self._event_library)
            self._event_library[name] = PulseqBlock(ID, rf, gx, gy, gz, adc, trig)
            self._event_id[name] = ID

    def section(self, section_name: str):
        if self._format == "pulseq":
            _current_seqlength = len(self._sequence.block_events)
        else:
            _current_seqlength = len(self._loop)
        self._sections_edges.append(_current_seqlength)            
        
        
    def add_block(
        self,
        event: str,
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
        if self._format == "pulseq":
            current_event = copy(self.event_library[event])

            # scale RF pulse and apply phase modulation / frequency offset
            if current_event["rf"] is not None:
                current_event["rf"].signal *= rf_amp
                current_event["rf"].phase_offset += rf_phase
                current_event["rf"].freq_offset += rf_freq

            # apply phase modulation to ADC
            if current_event["adc"] is not None:
                current_event["adc"].phase_offset += adc_phase

            # scale gradients
            if current_event["gx"] is not None:
                current_event["gx"] = pp.scale_grad(
                    grad=current_event["gx"], scale=gx_amp
                )
            if current_event["gy"] is not None:
                current_event["gy"] = pp.scale_grad(
                    grad=current_event["gy"], scale=gy_amp
                )
            if current_event["gz"] is not None:
                current_event["gz"] = pp.scale_grad(
                    grad=current_event["gz"], scale=gy_amp
                )

            # rotate gradients
            if rotmat is not None:
                # extract gradient waveforms from current event
                current_grad = {}
                for ch in ["gx", "gy", "gz"]:
                    if ch in current_event:
                        current_grad[ch] = current_event[ch]

                # actual rotation
                current_grad = _pp_rotate(current_grad, rotmat)

                # replace rotated gradients in current event
                for ch in ["gx", "gy", "gz"]:
                    if ch in current_event:
                        current_event[ch] = current_grad[ch]

            # update sequence structure
            if delay is None:
                self._sequence.add_block(*current_event.values())
            else:
                self._sequence.add_block(pp.make_delay(delay), *current_event.values())

        else:
            parent_block_id = self._event_id[event]
            block_duration = self._event_duration[event]
            if delay is not None:
                block_duration += delay
            if rotmat is None:
                rotmat = np.eye(3, dtype=np.float32).ravel().tolist()
            else:
                rotmat = rotmat.ravel().tolist()
            loop_row = [
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
            ] + rotmat
            self._loop.append(loop_row)


    def export(self):
        if self._format == "pulseq":
            return self._sequence

        # prepare Ceq structure
        self._sequence = Ceq(
            list(self._event_library.values()),
            self._loop,
            self._sections_edges,
        )

        # if required, dump to bytes string
        if self._format == "bytes":
            self._sequence = self._sequence.to_bytes()

        return self._sequence


def _pp_rotate(grad, rot_matrix):
    grad_channels = ["gx", "gy", "gz"]
    grad = copy.deepcopy(grad)

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
