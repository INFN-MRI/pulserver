"""Intermediate sequence representation."""

__all__ = ["Sequence"]

from copy import copy
from typing import SimpleNamespace

# import numpy as np
from ._ceq import Ceq


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
            import pypulseq as pp

            self._sequence = pp.Sequence(system=system)
        else:
            self._sequence = Ceq()

        self._event_library = {}
        self._event_id = {}
        self._n_sections = 1
        self._current_section = 0
        self._section_label = []
        self._section_library = None

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
        self._event_library[name] = {
            "rf": rf,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "adc": adc,
            "trig": trig,
        }
        self._event_id[name] = len(self._event_library)

    def section(self, section_name: str):
        if self._section_library is None:  # first section
            self._section_library = {section_name: self._current_section}
        elif section_name in self._section_library:  # pre-existing section
            self._current_section = self._section_library[section_name]
        else:  # new section
            self._current_section = self._n_sections
            self._n_sections += 1
            self._section_library[section_name] = self._current_section

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
        # rotmat: np.ndarray | None = None,
    ):
        if self._format == "pulseq":
            import pypulseq as pp

            current_event = copy(self.event_library[event])
            if current_event["rf"] is not None:
                current_event["rf"].signal *= rf_amp
                current_event["rf"].phase_offset += rf_phase
                current_event["rf"].freq_offset += rf_freq
            if current_event["adc"] is not None:
                current_event["adc"].phase_offset += adc_phase
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
            if delay is None:
                self._sequence.add_block(*current_event.values())
            else:
                self._sequence.add_block(pp.make_delay(delay), *current_event.values())
        else:
            ID = self._event_id[event]
            block_duration = self._event_duration[event]
            if delay is not None:
                block_duration += delay
            loop_row = [
                -1,
                ID,
                rf_amp,
                rf_phase,
                rf_freq,
                gx_amp,
                gy_amp,
                gz_amp,
                adc_phase,
                block_duration,
            ]
            self._sequence.loop.append(loop_row)
            self._section_label.append(self._current_section)
