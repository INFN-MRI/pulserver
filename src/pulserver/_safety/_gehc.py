"""
"""

__all__ = []

from types import SimpleNamespace

import numpy as np

from pypulseq import Opts

from .._core._ceq import PulseqRF, PulseqGrad, PulseqShapeTrap

SEGMENT_RINGDOWN_TIME = 116 * 1e-6  # TODO: doesn't have to be hardcoded


def compute_max_energy(ceq, window_width=10.0, windows_stride=5.0):

    # get block id
    block_id = ceq.loop[:, 1].astype(int)

    # get RF blocks
    _rf_blocks = [
        n for n in range(ceq.n_parent_blocks) if ceq.parent_blocks[n].rf is not None
    ]
    has_rf = np.stack([block_id == idx for idx in _rf_blocks]).sum(axis=0).astype(bool)

    # get segment id
    segment_id = np.ascontiguousarray(ceq.loop[:, 0])
    seg_boundaries = np.diff(segment_id + 1) != 0
    seg_boundaries = np.concatenate((seg_boundaries, np.asarray([True])))

    # get block duration
    block_dur = np.ascontiguousarray(ceq.loop[:, 9])
    block_dur[seg_boundaries] += SEGMENT_RINGDOWN_TIME  # add segment ringdown

    # get absolute sequence time axis
    block_starts = np.cumsum(np.concatenate(([0.0], block_dur)))[:-1]
    block_ends = np.concatenate((block_starts[1:], [block_starts[-1] + block_dur[-1]]))

    # get sequence end
    sequence_end = block_ends[-1]

    # get windows starts
    window_starts = np.arange(
        0, sequence_end - window_width + windows_stride, windows_stride
    )
    window_ends = window_starts + window_width
    window_ends[-1] = min(window_ends[-1], sequence_end)

    # loop over windows
    n_windows = len(window_starts)
    for n in range(n_windows):
        first_block = np.argmin(abs(block_starts - window_starts[n]))
        last_block = np.argmin(abs(block_ends - window_ends[n]))

        # get current blocks
        current_starts = block_starts[first_block:last_block][has_rf[first_block:last_block]]
        current_ends = block_starts[first_block:last_block][has_rf[first_block:last_block]]
        current_blocks = block_id[first_block:last_block][has_rf[first_block:last_block]]


def compute_rf_energy_with_raster(
    rf_raster_time,
    rf_blocks,
    rf_waveforms,
    scaling_factors,
    block_starts,
    window_size=10,
    stride=5,
):
    """
    Computes the maximum RF energy deposition in a sliding window of a given size with a specified stride,
    using a known RF raster time and RF pulse waveforms for specific blocks. The scaling factor for each
    RF execution is a scalar.

    Parameters
    ----------
    rf_raster_time : float
        The time resolution or sampling interval of the RF pulses (raster time).

    rf_blocks : list of int
        A list containing the block indices where RF pulses are present. Each index corresponds to a block with RF.

    rf_waveforms : list of 1D numpy arrays
        A list of RF pulse waveforms (amplitude envelopes) for each block in `rf_blocks`.

    scaling_factors : list of floats
        A list containing scalar scaling factors applied to each corresponding RF waveform.

    block_starts : 1D numpy array
        A 1D numpy array containing the start times of each block in the sequence.

    window_size : float, optional, default=10
        Size of the sliding window in seconds.

    stride : float, optional, default=5
        The stride or step size for the sliding window in seconds.

    Returns
    -------
    max_energy : float
        The maximum RF energy deposition in any sliding window.

    window_energies : 1D numpy array
        The RF energy deposition values for each sliding window.

    window_starts : 1D numpy array
        The starting times of each sliding window.

    Notes
    -----
    - The RF energy deposition in each window is computed as the sum of the squared waveform amplitudes
      (accounting for scalar scaling factors) integrated over time using the RF raster time.
    - This function assumes the waveform is provided with an RF raster time, simplifying the time calculations.
    """
    # Step 1: Prepare a timeline covering the entire sequence duration
    sequence_end = block_starts[-1] + (len(rf_waveforms[-1]) * rf_raster_time)

    # Create sliding window start times
    window_starts = np.arange(0, sequence_end - window_size + stride, stride)

    # Initialize an array to store the energy in each window
    window_energies = np.zeros_like(window_starts)

    # Step 2: Calculate energy for each sliding window
    for i, window_start in enumerate(window_starts):
        window_end = window_start + window_size

        # Initialize energy for this window
        total_energy = 0.0

        # Loop through RF blocks
        for block_idx, waveform, scaling, block_start in zip(
            rf_blocks, rf_waveforms, scaling_factors, block_starts[rf_blocks]
        ):
            block_duration = len(waveform) * rf_raster_time
            block_end = block_start + block_duration

            # Check if the block overlaps with the current window
            if block_end <= window_start or block_start >= window_end:
                continue  # Skip if block is outside the window

            # Find the overlap between the block and the current window
            overlap_start = max(block_start, window_start)
            overlap_end = min(block_end, window_end)

            # Compute indices corresponding to the overlap within the waveform
            start_idx = int((overlap_start - block_start) / rf_raster_time)
            end_idx = int((overlap_end - block_start) / rf_raster_time)

            # Get the overlapping portion of the waveform
            selected_waveform = waveform[start_idx:end_idx]

            # Apply the scalar scaling factor to the waveform
            scaled_waveform = scaling * selected_waveform

            # Calculate energy as the sum of squared amplitude in the overlapping section
            energy_contribution = np.sum(scaled_waveform**2) * rf_raster_time
            total_energy += energy_contribution

        # Store the total energy in this window
        window_energies[i] = total_energy

    # Step 3: Find the maximum energy deposition across all windows
    max_energy = np.max(window_energies)

    return max_energy, window_energies, window_starts


def _rfstat(rf: PulseqRF, system: Opts) -> SimpleNamespace:

    # get waveform in physical units
    wave_max = abs(rf.wav.magnitude).max()
    waveform = rf.wav.amplitude * (rf.wav.magnitude / wave_max)

    # add phase
    if rf.wav.phase is not None:
        waveform *= np.exp(1j * rf.wav.phase)

    # convert extended trapezoid to arbitrary
    if rf.type == 1:
        waveform, time = waveform, rf.wav.time + rf.delay
    elif rf.type == 2:
        waveform, time = _extended2arb(
            waveform, rf.wav.time, system.rf_raster_time, rf.delay
        )

    return SimpleNamespace(waveform=waveform, time=time, raster=system.rf_raster_time)


def _gradstat(grad: PulseqGrad, system: Opts):

    # get waveform in physical units
    if grad.type == 1:
        waveform = grad.trap
    else:
        wave_max = abs(grad.shape.magnitude).max()
        waveform = grad.shape.amplitude * (grad.shape.magnitude / wave_max)

    # convert extended trapezoid to arbitrary
    if grad.type == 1:
        waveform, time = _trap2arb(waveform, system.grad_raster_time, grad.delay)
    elif grad.type == 2:
        waveform, time = waveform, grad.shape.time + grad.delay
    elif grad.type == 3:
        waveform, time = _extended2arb(
            waveform, grad.shape.time, system.grad_raster_time, grad.delay
        )

    return SimpleNamespace(waveform=waveform, time=time, raster=system.grad_raster_time)


# %% local subroutines
def _trap2arb(trap: PulseqShapeTrap, dt: float, delay: float) -> np.ndarray:
    waveform, time = _trap2extended(trap)
    return _extended2arb(waveform, time, dt, delay)


def _trap2extended(trap):
    if trap.flat_time > 0:
        waveform = np.asarray([0, 1, 1, 0]) * trap.amplitude
        time = np.asarray(
            [
                0,
                trap.rise_time,
                trap.rise_time + trap.flat_time,
                trap.rise_time + trap.flat_time + trap.fall_time,
            ]
        )
    else:
        waveform = np.asarray([0, 1, 0]) * trap.amplitude
        time = np.asarray([0, trap.rise_time, trap.rise_time + trap.fall_time])

    return waveform, time


def _extended2arb(
    waveform: np.ndarray, time: np.ndarray, dt: float, delay: float
) -> np.ndarray:

    _waveform = waveform
    _time = delay + time

    if delay > 0:
        _waveform = np.concatenate(([0], _waveform))
        _time = np.concatenate(([0], _time))

    time = _arange(0.5 * dt, _time[-1], dt)
    return np.interp(time, _time, _waveform, left=0, right=0), time


def _arange(start, stop, step=1):
    if stop is None:
        stop = step
        step = 1

    tol = 2.0 * np.finfo(float).eps * max(abs(start), abs(stop))
    sig = np.sign(step)

    # Exceptional cases
    if not np.isfinite(start) or not np.isfinite(step) or not np.isfinite(stop):
        return np.array([np.nan])
    elif step == 0 or (start < stop and step < 0) or (stop < start and step > 0):
        # Result is empty
        return np.zeros(0)

    # n = number of intervals = length(v) - 1
    if start == np.floor(start) and step == 1:
        # Consecutive integers
        n = int(np.floor(stop) - start)
    elif start == np.floor(start) and step == np.floor(step):
        # Integers with spacing > 1
        q = np.floor(start / step)
        r = start - q * step
        n = int(np.floor((stop - r) / step) - q)
    else:
        # General case
        n = round((stop - start) / step)
        if sig * (start + n * step - stop) > tol:
            n -= 1

    # last = right hand end point
    last = start + n * step
    if sig * (last - stop) > -tol:
        last = stop

    # out should be symmetric about the mid-point
    out = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    out[k] = start + k * step
    out[n - k] = last - k * step
    if n % 2 == 0:
        out[n // 2 + 1] = (start + last) / 2

    return out
