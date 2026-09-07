#!/usr/bin/env python3
"""Background figures for the mechanical-resonance page.

The mode itself: what a mechanical resonance answers to in frequency, and how
long it takes to answer. Both panels are drawn from the damped-oscillator
response at a representative quality factor, in units of the mode's own centre
frequency and memory, so nothing here depends on a particular coil.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pulserver.pypulseq as pp
from _figures import FAINT, INK, MUTED, SERIES, _style

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[0] / "explanations" / "assets" / "mechanical_resonance"

#: Quality factor the panels are drawn at: high enough that the band is
#: visibly narrow, low enough that the build-up fits on one screen.
Q = 12.0


#: The window the page is drawn at, in seconds.
REFERENCE_WINDOW_S = 0.020
#: Gradient raster the elements are rendered on, in seconds.
RASTER_S = 4e-6
#: The frequency range every published band falls in, in Hz.
TERRITORY_HZ = (300.0, 3000.0)
GAMMA_HZ_PER_MT_PER_M = 42.576e6 * 1e-3


def _zoo(module: str, **kwargs):
    """One sequence from the shipped zoo."""
    from importlib import import_module

    system = kwargs.pop("system", None) or pp.Opts(
        max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s"
    )
    made = getattr(import_module("pulserver.app"), module).main(
        system=system, n_dummy=0, **kwargs
    )
    return made[0] if isinstance(made, tuple) else made


def _event_samples(grad, local):
    """One gradient event sampled on ``local``, its own time base in seconds."""
    if grad.type == "trap":
        edges = np.cumsum([grad.delay, grad.rise_time, grad.flat_time, grad.fall_time])
        times = np.concatenate([[0.0], edges])
        values = np.array([0.0, 0.0, grad.amplitude, grad.amplitude, 0.0])
        return np.interp(local, times, values, left=0.0, right=0.0)
    times = np.asarray(grad.tt, float) + float(grad.delay)
    values = np.asarray(grad.waveform, float)
    return np.interp(local, times, values, left=0.0, right=0.0)


def render_axis(sequence, axis: str, keep=None):
    """One physical axis of a sequence on the gradient raster, in mT/m.

    ``keep(index, block)`` selects which blocks contribute, so a single
    element of a sequence — an echo train, a blip train, a slice select — can
    be read on its own.
    """
    starts, total = [], 0.0
    for index in range(1, len(sequence.block_events) + 1):
        starts.append(total)
        total += float(sequence.get_block(index).block_duration)
    t = np.arange(0.0, total + RASTER_S, RASTER_S)
    g = np.zeros_like(t)
    for index in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(index)
        if keep is not None and not keep(index, block):
            continue
        grad = getattr(block, axis, None)
        if grad is None:
            continue
        start = starts[index - 1]
        span = (t >= start) & (t <= start + float(block.block_duration))
        g[span] += _event_samples(grad, t[span] - start)
    return t, g / GAMMA_HZ_PER_MT_PER_M


def sliding_equivalent_sinusoid(t, g, freqs, window: float = REFERENCE_WINDOW_S):
    """The largest equivalent sinusoid any window of ``window`` sees, per
    frequency, in the units ``g`` is given in.

    The waveform is padded with one silent window at each end, so what comes
    back is what this element alone drives, wherever the window falls on it.
    """
    dt = float(t[1] - t[0])
    pad = round(window / dt)
    padded = np.concatenate([np.zeros(pad), np.asarray(g, float), np.zeros(pad)])
    times = np.arange(padded.size) * dt
    out = np.empty(len(freqs))
    for k, f in enumerate(freqs):
        x = padded * np.exp(-2j * np.pi * f * times)
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * dt)])
        out[k] = np.abs(cumulative[pad:] - cumulative[:-pad]).max() * 2.0 / window
    return out


def figure_mode() -> Path:
    """The two things that describe a mode: its band, and its memory."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.2), dpi=170)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.19, wspace=0.28)

    # (a) response against frequency, normalised to the mode's centre
    x = np.linspace(0.55, 1.45, 4000)
    response = 1.0 / np.sqrt((1.0 - x**2) ** 2 + (x / Q) ** 2)
    response /= response.max()
    half = 1.0 / Q  # the -3 dB width, in units of the centre frequency
    left.fill_between(
        [1 - half / 2, 1 + half / 2], 0, 1.08, color=SERIES[1], alpha=0.16, lw=0
    )
    left.plot(x, response, color=SERIES[0], lw=1.6)
    left.axhline(1 / np.sqrt(2), color=FAINT, lw=0.8, ls=(0, (4, 3)))
    left.annotate(
        "",
        xy=(1 - half / 2, 1 / np.sqrt(2)),
        xytext=(1 + half / 2, 1 / np.sqrt(2)),
        arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9},
    )
    left.text(1.0, 0.60, r"$\Delta f$", ha="center", fontsize=9, color=INK)
    left.text(
        1.0,
        1.12,
        "the forbidden band",
        ha="center",
        fontsize=8.5,
        color=SERIES[1],
    )
    left.set_xlim(0.55, 1.45)
    left.set_ylim(0, 1.3)
    left.set_yticks([0, 0.5, 1.0])
    left.set_xticks([1.0])
    left.set_xticklabels([r"$f_0$"])
    left.set_xlabel("drive frequency")
    left.set_ylabel("response (peak = 1)")
    _style(left, "(a) frequency response")

    # (b) build-up in time under a drive at the centre frequency
    t = np.linspace(0, 4.0, 6000)
    envelope = 1.0 - np.exp(-t)
    right.plot(t, envelope, color=SERIES[0], lw=1.4, zorder=3)
    right.plot(t, -envelope, color=SERIES[0], lw=1.4, zorder=3)
    right.plot(
        t,
        envelope * np.sin(2 * np.pi * (Q / np.pi) * t),
        color=SERIES[0],
        lw=0.6,
        alpha=0.5,
        zorder=2,
    )
    right.axvline(1.0, color=FAINT, lw=0.8, ls=(0, (4, 3)))
    right.text(1.06, -1.24, "one memory", fontsize=8.5, color=MUTED)
    right.plot([0.35], [1 - np.exp(-0.35)], "o", ms=4, color=SERIES[1], zorder=4)
    right.annotate(
        "a drive that stops here\nnever reaches full amplitude",
        xy=(0.35, 1 - np.exp(-0.35)),
        xytext=(1.35, 0.42),
        fontsize=8.5,
        color=SERIES[1],
        arrowprops={"arrowstyle": "-", "color": SERIES[1], "lw": 0.8},
    )
    right.set_xlim(0, 4.0)
    right.set_ylim(-1.45, 1.45)
    right.set_yticks([-1, 0, 1])
    right.set_xticks([0, 1, 2, 3, 4])
    right.set_xlabel("time (mode memory units)")
    right.set_ylabel("coil deformation")
    _style(right, "(b) build-up in time")

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "mode_response.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


#: Lobes per window in the equivalent-sinusoid figure: the drive period is
#: two lobes, so the fundamental sits at eight cycles per window.
LOBES_PER_WINDOW = 16
#: Fraction of a lobe spent on each ramp.
RAMP_FRACTION = 0.2


def _lobe_train(t, n_lobes: int, lobe: float):
    """A bipolar trapezoidal readout train of unit plateau, starting at t = 0.

    Alternating lobes of duration ``lobe`` with ramps at each end, which is an
    echo train with its blips and its phase encoding taken away.
    """
    out = np.zeros_like(t)
    ramp = RAMP_FRACTION * lobe
    for k in range(n_lobes):
        t0 = k * lobe
        local = t - t0
        inside = (local >= 0.0) & (local < lobe)
        shape = np.clip(np.minimum(local, lobe - local) / ramp, 0.0, 1.0)
        out = np.where(inside, (-1.0) ** k * shape, out)
    return out


def _equivalent_sinusoid(t, g, f, window: float) -> float:
    """The amplitude of the sinusoid at ``f`` carrying ``g``'s Fourier content
    over one window: the definition the check evaluates."""
    phasor = np.exp(-2j * np.pi * f * t)
    return 2.0 / window * abs(np.trapezoid(g * phasor, t))


def figure_equivalent_sinusoid() -> Path:
    """What a train's length does to the sinusoid it is equivalent to."""
    window = 1.0
    lobe = window / LOBES_PER_WINDOW
    f0 = 0.5 / lobe  # the train alternates, so its period is two lobes
    t = np.linspace(0.0, window, 200_001)

    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.3), dpi=170)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.19, wspace=0.26)

    shown = (2, 4, 8, 16)
    ticks, labels = [], []
    for row, n_lobes in enumerate(reversed(shown)):
        g = _lobe_train(t, n_lobes, lobe)
        offset = 2.6 * row
        left.plot(t / window, g + offset, color=SERIES[0], lw=1.0)
        left.axhline(offset, color=FAINT, lw=0.6, zorder=0)
        ticks.append(offset)
        labels.append(f"{n_lobes * lobe / window:.2f} W")
    left.set_xlim(0, 1.0)
    left.set_ylim(-1.6, 2.6 * len(shown))
    left.set_yticks(ticks)
    left.set_yticklabels(labels)
    left.set_ylabel("train length")
    left.set_xticks([0, 0.5, 1.0])
    left.set_xticklabels(["0", "W/2", "W"])
    left.set_xlabel("time (one window)")
    _style(left, "(a) gradient temporal envelope")

    fractions = np.linspace(0.05, 2.0, 60)
    readings = []
    for fraction in fractions:
        n_lobes = max(1, round(fraction * window / lobe))
        readings.append(
            _equivalent_sinusoid(t, _lobe_train(t, n_lobes, lobe), f0, window)
        )
    fractions = np.array(
        [max(1, round(f * window / lobe)) * lobe / window for f in fractions]
    )
    right.plot(fractions, readings, color=SERIES[0], lw=1.5)
    saturated = max(readings)
    right.axhline(saturated, color=FAINT, lw=0.8, ls=(0, (4, 3)))
    right.text(
        0.06,
        saturated + 0.04,
        "what a train that outlasts the window sustains",
        fontsize=8.5,
        color=MUTED,
    )
    right.axvline(1.0, color=FAINT, lw=0.8, ls=(0, (4, 3)))
    right.text(1.04, 0.06, "one window", fontsize=8.5, color=MUTED)
    for fraction in (0.125, 0.25, 0.5, 1.0):
        n_lobes = round(fraction * window / lobe)
        value = _equivalent_sinusoid(t, _lobe_train(t, n_lobes, lobe), f0, window)
        right.plot([fraction], [value], "o", ms=4, color=SERIES[1], zorder=3)
    right.set_xlim(0, 2.0)
    right.set_ylim(0, saturated * 1.28)
    right.set_xlabel("train length (windows)")
    right.set_ylabel("mechanical response (plateau = 1)")
    _style(right, "(b) mechanical response at $f_0$")

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "equivalent_sinusoid.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def _gallery(cases: list[dict], out_name: str) -> Path:
    """Two elements side by side: the gradient in time, and what it drives.

    Both drive panels share one vertical scale, because the whole point of a
    pair is which of the two the coil can hear.
    """
    freqs = np.linspace(TERRITORY_HZ[0], TERRITORY_HZ[1], 541)
    for case in cases:
        case["drive"] = sliding_equivalent_sinusoid(case["t"], case["g"], freqs)

    ceiling = max(case["drive"].max() for case in cases) * 1.3
    tallest = max(np.abs(case["g"]).max() for case in cases) * 1.15
    figure, axes = plt.subplots(2, len(cases), figsize=(8.6, 4.6), dpi=170)
    figure.subplots_adjust(
        left=0.09, right=0.98, top=0.87, bottom=0.11, wspace=0.22, hspace=0.62
    )
    letters = "abcd"
    for column, case in enumerate(cases):
        top, bottom = axes[0][column], axes[1][column]
        lo, hi = case["zoom"]
        span = (case["t"] >= lo) & (case["t"] <= hi)
        top.plot((case["t"][span] - lo) * 1e3, case["g"][span], color=SERIES[0], lw=1.0)
        top.axhline(0.0, color=FAINT, lw=0.6, zorder=0)
        top.set_xlabel("ms")
        if column == 0:
            top.set_ylabel("gradient (mT/m)")
        top.set_ylim(-tallest, tallest)
        _style(top, "")
        top.set_title(
            f"({letters[column]}) {case['title']}",
            loc="left",
            fontsize=9,
            color=INK,
            pad=8,
        )

        peak = case["drive"].argmax()
        bottom.plot(freqs, case["drive"], color=SERIES[0], lw=1.2)
        bottom.plot([freqs[peak]], [case["drive"][peak]], "o", ms=4, color=SERIES[1])
        bottom.annotate(
            f"{case['drive'][peak]:.2f} mT/m\nat {freqs[peak]:.0f} Hz",
            xy=(freqs[peak], case["drive"][peak]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=SERIES[1],
        )
        bottom.set_ylim(0, ceiling)
        bottom.set_xlim(*TERRITORY_HZ)
        bottom.set_xlabel("frequency (Hz)")
        if column == 0:
            bottom.set_ylabel("mechanical response (mT/m)")
        _style(bottom, "")
        bottom.set_title(
            f"({letters[column + len(cases)]}) mechanical response",
            loc="left",
            fontsize=9,
            color=INK,
            pad=8,
        )

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / out_name
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def figure_epi() -> Path:
    """The two gradients an echo-planar train plays, read the same way."""
    seq = _zoo("epi2D_sequence", n_x=64, n_y=64, readout_bandwidth_hz=250e3)

    def acquiring(_index, block):
        return block.adc is not None

    t, readout = render_axis(seq, "gx", acquiring)
    _, blips = render_axis(seq, "gy", acquiring)
    return _gallery(
        [
            {
                "title": "readout gradient",
                "t": t,
                "g": readout,
                "zoom": (3.5e-3, 8.0e-3),
            },
            {
                "title": "phase-encode blips",
                "t": t,
                "g": blips,
                "zoom": (3.5e-3, 8.0e-3),
            },
        ],
        "elements_epi.png",
    )


def figure_spiral() -> Path:
    """A sweep crosses a band once: how slowly decides what it leaves there."""
    weak = pp.Opts(max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    long_arm = _zoo("gre_spiral2D_sequence", system=weak, n_x=128, n_arms=1, tr=None)
    short_arm = _zoo("gre_spiral2D_sequence", n_x=128, n_arms=16, tr=None)

    def acquiring(_index, block):
        return block.adc is not None and _index < 40

    t_long, g_long = render_axis(long_arm, "gx", acquiring)
    t_short, g_short = render_axis(short_arm, "gx", acquiring)
    span_long = t_long[np.abs(g_long) > 0]
    span_short = t_short[np.abs(g_short) > 0]
    return _gallery(
        [
            {
                "title": "one long arm",
                "t": t_long,
                "g": g_long,
                "zoom": (span_long.min(), span_long.max()),
            },
            {
                "title": "sixteen short arms",
                "t": t_short,
                "g": g_short,
                "zoom": (span_short.min(), span_short.min() + 0.02),
            },
        ],
        "elements_spiral.png",
    )


def window_profile(t, g, f: float, window: float = REFERENCE_WINDOW_S):
    """The equivalent sinusoid at ``f`` as the window slides, start by start.

    Returns the window start times and the reading at each, in the units ``g``
    is given in. The verdict is the largest of these; the profile shows how
    much placement decides.
    """
    dt = float(t[1] - t[0])
    span = round(window / dt)
    x = np.asarray(g, float) * np.exp(-2j * np.pi * f * np.asarray(t, float))
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * dt)])
    values = np.abs(cumulative[span:] - cumulative[:-span]) * 2.0 / window
    return t[: values.size], values


def figure_window_placement() -> Path:
    """The window slides over the scan, and the verdict is its worst position."""
    weak = pp.Opts(max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    seq = _zoo("gre_spiral2D_sequence", system=weak, n_x=128, n_arms=1, tr=None)

    def acquiring(_index, block):
        return block.adc is not None

    t, g = render_axis(seq, "gx", acquiring)
    freqs = np.linspace(TERRITORY_HZ[0], TERRITORY_HZ[1], 541)
    drive = sliding_equivalent_sinusoid(t, g, freqs)
    f0 = freqs[drive.argmax()]
    starts, profile = window_profile(t, g, f0)
    best = int(np.argmax(profile))

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 4.6), dpi=170, sharex=True, height_ratios=(1.0, 1.1)
    )
    figure.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.13, hspace=0.42)

    top.plot(t * 1e3, g, color=SERIES[0], lw=0.5)
    tallest = np.abs(g).max() * 1.45
    shown = (starts[0], starts[starts.size // 2], starts[best])
    for start in shown:
        loud = start == starts[best]
        top.add_patch(
            plt.Rectangle(
                (start * 1e3, -tallest),
                REFERENCE_WINDOW_S * 1e3,
                2 * tallest,
                facecolor=SERIES[1] if loud else MUTED,
                alpha=0.18 if loud else 0.09,
                lw=0,
                zorder=0,
            )
        )
    top.text(
        (starts[best] + REFERENCE_WINDOW_S / 2) * 1e3,
        tallest * 0.74,
        "loudest placement",
        ha="center",
        fontsize=8.5,
        color=SERIES[1],
    )
    top.set_ylim(-tallest, tallest)
    top.set_ylabel("gradient (mT/m)")
    _style(top, "(a) gradient temporal envelope")

    bottom.plot(starts * 1e3, profile, color=SERIES[0], lw=1.3)
    bottom.plot([starts[best] * 1e3], [profile[best]], "o", ms=4.5, color=SERIES[1])
    bottom.annotate(
        f"the verdict: {profile[best]:.1f} mT/m",
        xy=(starts[best] * 1e3, profile[best]),
        xytext=(10, -4),
        textcoords="offset points",
        fontsize=8.5,
        color=SERIES[1],
    )
    bottom.set_ylim(0, profile.max() * 1.3)
    bottom.set_xlabel("window start (ms)")
    bottom.set_ylabel(f"response at {f0:.0f} Hz (mT/m)")
    _style(bottom, "(b) mechanical response")

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "window_placement.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def main() -> int:
    print("wrote", figure_mode())
    print("wrote", figure_equivalent_sinusoid())
    print("wrote", figure_epi())
    print("wrote", figure_spiral())
    print("wrote", figure_window_placement())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
