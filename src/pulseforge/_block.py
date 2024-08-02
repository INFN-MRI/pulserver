"""Pulseq Block structure."""

__all__ = ["PulseqBlock"]


class PulseqBlock:
    """
    Pulseq Block helper.

    This is related to Pulceq PulseqBlock structure.

    Each block is a collection of one or more PyPulseq events. For each PulseqBlock,
    a maximum of 1 event for each board (rf, gx, gy, gz, adc, trig) can be executed.

    """

    def __init__(self):
        self._event_library = {}

    def add_block(self, name, rf=None, gx=None, gy=None, gz=None, adc=None, trig=None):
        if rf is not None and adc is not None:
            VALID_BLOCK = False
        else:
            VALID_BLOCK = True
        assert VALID_BLOCK, "Error! A block cannot contain both a RF and ADC event."

        self._event_library[name] = {
            "rf": rf,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "adc": adc,
            "trig": trig,
        }

    def __call__(self, seq, *args, **kwargs):
        pass
