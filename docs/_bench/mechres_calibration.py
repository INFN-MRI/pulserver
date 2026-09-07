"""Lockout tables, design helpers and two figures for the mechanical-resonance pages.

The tables a gradient coil ships are read here into forbidden bands; the
sequence zoo is driven from here into designs with a wanted echo spacing or
readout plateau; and a design is read here into the amplitude the check
would judge. The calibration corpus that uses all of this is
``mechres_corpus.py``.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pulserver.pypulseq as pp
from _figures import FAINT, INK, SERIES, _style

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[0] / "explanations" / "assets" / "mechanical_resonance"
SCALE = HERE / "mechres_scale.json"
LOCKOUT_DIRS = [
    Path.home() / "pulserver-project" / "lockout",
    Path.home() / "PulseStudio" / "efgre3d",
]

GAMMA_HZ_PER_T = 42.576e6
GAMMA_HZ_PER_MT_PER_M = GAMMA_HZ_PER_T * 1e-3
#: ``SA_ZERO_BAND_SINUSOID_MT_PER_M``: drawn, never recomputed here.
FLOOR_MT_PER_M = 10.0
#: ``SA_AEQ_TRAIN_SHAPE``: the fundamental of a triangular lobe train as a
#: fraction of its plateau, which is how a stated plateau becomes a sinusoid.
TRAIN_SHAPE = 8.0 / math.pi**2
#: The readout plateau a design is aimed at, in mT/m: the amplitude the one
#: tolerance the tables state is expressed as. Aiming at the gradient limit
#: instead would put prescriptions in the corpus that no console offers.
TARGET_PLATEAU_MT_PER_M = 16.0
#: The range every inspected band falls in, read densely on every axis.
TERRITORY = (300.0, 3000.0)
#: A tolerance nothing sustains under: every band is refused on the bound,
#: so what comes back is the scan's own reading and its contributors.
TINY_HZ_PER_M = 1.0
#: Raster times the lockout tables were written against. A coarser raster
#: snaps the dwell and hides receiver bandwidths a scanner can really
#: prescribe, which shrinks the design space the corpus can explore.
RASTERS = {
    "grad_raster_time": 4e-6,
    "adc_raster_time": 2e-6,
    "rf_raster_time": 2e-6,
    "block_duration_raster": 4e-6,
}
SYSTEM = pp.Opts(
    max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s", **RASTERS
)
WEAK = pp.Opts(
    max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", **RASTERS
)
#: Designs a parameter sweep could not build, reported at the end.
INFEASIBLE: list[tuple] = []

plt.rcParams.update(
    {
        "font.size": 11.0,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.0,
    }
)

# ----------------------------------------------------------------------
# The vendor's tables


def _numbers(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        try:
            rows.append([float(x) for x in body.split()])
        except ValueError:
            continue
    return rows


def read_epi_table(path: Path) -> dict:
    """One echo-spacing lockout table as bands per axis.

    A row is an echo-spacing range in microseconds and a tolerance in G/cm;
    the band it guards is the train fundamental, f = 1 / (2 ESP).
    """
    text = path.read_text().splitlines()
    bands = {"x": [], "y": [], "z": []}
    axis = None
    pending = None
    for line in text:
        head = line.strip().lower()
        if head.startswith("#"):
            for name in ("x", "y", "z"):
                if head.startswith(f"# {name} axis"):
                    axis = name
                    pending = None
            continue
        parts = head.split()
        if axis is None or not parts:
            continue
        if pending is None and len(parts) == 1:
            pending = int(float(parts[0]))
            continue
        if pending and len(parts) == 3:
            lo_us, hi_us, tol = (float(p) for p in parts)
            bands[axis].append(
                {
                    "esp_us": (lo_us, hi_us),
                    "f_hz": (1e6 / (2.0 * hi_us), 1e6 / (2.0 * lo_us)),
                    "tolerance_mt_per_m": 10.0 * tol,  # G/cm -> mT/m
                }
            )
            pending -= 1
    return {"file": path.name, "bands": bands}


def _name_coils(tables: dict) -> None:
    """Give every coil in the tables a letter, in a stable order."""
    names = sorted({coil_of(t["file"]) for t in tables["epi"]})
    COIL_ALIASES.clear()
    for index, name in enumerate(names):
        COIL_ALIASES[name] = chr(ord("A") + index)


def load_tables() -> dict:
    epi = []
    for folder in LOCKOUT_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("epiesp*.dat")):
            epi.append(read_epi_table(path))
    tables = {"epi": epi}
    _name_coils(tables)
    return tables


# ----------------------------------------------------------------------
# Reading a sequence the way the gate does


def _sequence(result):
    return result[0] if isinstance(result, tuple) else result


def dense_reading(sequence, memory: float | None = None) -> dict:
    """The exact window reading of a sequence over the territory, per axis.

    One wide band per axis at a tolerance nothing sustains under, so every
    band is refused and what comes back is the scan read exactly, not the
    bound over its repetitions.
    """
    bands = [(TERRITORY[0], TERRITORY[1], TINY_HZ_PER_M, axis) for axis in "xyz"]
    overlay = sequence.calculate_gradient_spectrum(
        plot=False,
        max_frequency=TERRITORY[1],
        tr="worst_case",
        resonance_lines=True,
        bands=bands,
        memory=memory,
    )[4]
    freqs = np.asarray(overlay.candidate_freqs, float)
    amps = np.asarray(overlay.candidate_a_eq, float) / GAMMA_HZ_PER_MT_PER_M
    return {"freqs": freqs, "amps": amps}


def in_band(reading: dict, band: tuple[float, float], axis: int) -> float:
    inside = (reading["freqs"] >= band[0]) & (reading["freqs"] <= band[1])
    return float(reading["amps"][inside, axis].max()) if inside.any() else 0.0


# ----------------------------------------------------------------------
# The scenarios


def _zoo(module, system=SYSTEM, **kwargs):
    from importlib import import_module

    return _sequence(
        getattr(import_module("pulserver.app"), module).main(
            system=system, n_dummy=0, **kwargs
        )
    )


def _acquisition_starts_and_durations_s(sequence) -> tuple[list[float], list[float]]:
    """Start time and duration of every block that acquires, in order."""
    starts, durations, t = [], [], 0.0
    for index in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(index)
        if block.adc is not None:
            starts.append(t)
            durations.append(float(block.block_duration))
        t += float(block.block_duration)
    return starts, durations


def _acquisition_durations_s(sequence) -> list[float]:
    return _acquisition_starts_and_durations_s(sequence)[1]


def _echo_spacing_s(sequence) -> float:
    """The period of the acquisitions that repeat back to back: the most
    common spacing between consecutive acquisition starts, whatever blocks
    lie between them."""
    starts, _ = _acquisition_starts_and_durations_s(sequence)
    if len(starts) < 2:
        return float("nan")
    gaps = np.diff(np.asarray(starts))
    gaps = gaps[gaps <= 3.0 * np.median(gaps)]  # within a train, not across repetitions
    values, counts = np.unique(np.round(gaps, 7), return_counts=True)
    return float(values[np.argmax(counts)])


def _repetition_time_s(sequence) -> float:
    """The spacing between one repetition's acquisitions and the next: the
    largest common gap between acquisition starts."""
    starts, _ = _acquisition_starts_and_durations_s(sequence)
    if len(starts) < 2:
        return float("nan")
    gaps = np.diff(np.asarray(starts))
    values, counts = np.unique(np.round(gaps, 7), return_counts=True)
    return float(values[np.argmax(counts)])


def _readout_duration_s(sequence) -> float:
    durations = _acquisition_durations_s(sequence)
    return max(durations) if durations else float("nan")


def _readout_plateau_mt_per_m(sequence) -> float:
    """The largest gradient amplitude while the ADC is sampling.

    A stated tolerance is the readout's flat top in G/cm, so this is measured
    over the sampling interval alone: a prewinder or a flyback lobe that
    shares the acquisition block is not the plateau.
    """
    peak = 0.0
    for index in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(index)
        if block.adc is None:
            continue
        start = float(block.adc.delay)
        stop = start + float(block.adc.num_samples) * float(block.adc.dwell)
        local = np.linspace(start, stop, 64)
        for grad in (block.gx, block.gy, block.gz):
            if grad is None:
                continue
            if grad.type == "trap":
                edges = np.cumsum(
                    [grad.delay, grad.rise_time, grad.flat_time, grad.fall_time]
                )
                times = np.concatenate([[0.0], edges])
                values = np.array([0.0, 0.0, grad.amplitude, grad.amplitude, 0.0])
            else:
                times = np.asarray(grad.tt, float) + float(grad.delay)
                values = np.asarray(grad.waveform, float)
            sampled = np.interp(local, times, values, left=0.0, right=0.0)
            peak = max(peak, float(np.abs(sampled).max()))
    return peak / GAMMA_HZ_PER_MT_PER_M


def _multiecho(n_x: int, bandwidth: float, n_echoes: int, monopolar: bool):
    """One multi-echo design, or ``None`` when the system cannot play it."""
    try:
        return _zoo(
            "gre_multiecho2D_sequence",
            n_x=n_x,
            # Enough repetitions for the longest window to be full, no more.
            n_y=12,
            n_echoes=n_echoes,
            monopolar=monopolar,
            readout_bandwidth_hz=float(bandwidth),
        )
    except Exception as exc:
        INFEASIBLE.append(("gre_multiecho2D_sequence", float(bandwidth), str(exc)[:60]))
        return None


MULTIECHO_MATRICES = (32, 48, 64, 96, 128, 160, 192, 256)


def multiecho_drive_hz(esp_s: float, monopolar: bool) -> float:
    """The frequency a multi-echo train drives at its echo spacing.

    A flyback train repeats with the spacing; a bipolar train alternates sign,
    so it repeats with twice the spacing and drives half the frequency. The
    other of the two is a harmonic the train's own symmetry suppresses.
    """
    return (1.0 if monopolar else 0.5) / esp_s


def multiecho_with_spacing_in(
    esp_s: tuple[float, float], n_echoes: int, monopolar: bool
):
    """Every multi-echo train whose echo spacing lands inside ``esp_s``.

    The spacing is the readout duration plus a fixed rewind, so it steps with
    the dwell: ``n_x`` samples move it in jumps the bandwidth cannot subdivide,
    and a sweep lands on a sparse set rather than anywhere asked for. Each
    matrix is probed once to learn its rewind, then asked directly for the
    bandwidth that gives a wanted spacing. Returns ``(sequence, spacing,
    bandwidth)`` per distinct spacing reached, loudest plateau first.
    """
    lo, hi = esp_s
    found = {}
    for n_x in MULTIECHO_MATRICES:
        probe = _multiecho(n_x, 100e3, n_echoes, monopolar)
        if probe is None:
            continue
        esp = _echo_spacing_s(probe)
        if not np.isfinite(esp):
            continue
        rewind = esp - n_x / 100e3
        for target in np.linspace(lo, hi, 17):
            read = target - rewind
            if read <= 0.0:
                continue
            seq = _multiecho(n_x, n_x / read, n_echoes, monopolar)
            if seq is None:
                continue
            esp = _echo_spacing_s(seq)
            if not np.isfinite(esp) or not (lo <= esp <= hi):
                continue
            plateau = _readout_plateau_mt_per_m(seq)
            key = round(esp * 1e6)
            if key not in found or plateau > found[key][0]:
                found[key] = (plateau, (seq, esp, n_x / read))
    return [v[1] for v in sorted(found.values(), key=lambda v: -v[0])]


def coil_of(name: str) -> str:
    """The gradient coil an echo-spacing table file belongs to, from its name."""
    stem = name.replace("epiesp", "").replace(".dat", "")
    stem = stem.lstrip(".").split(".")[0] or "default"
    return stem.upper()


def vendor_bands(tables: dict, coils: set | None = None) -> list[dict]:
    """Every band the echo-spacing tables guard, per axis, each with the
    readout plateau it tolerates; with ``coils`` only the tables of those
    coils. These are the only bands any family is judged against."""
    out = []

    def wanted(name: str) -> bool:
        return coils is None or coil_of(name) in coils

    for table in tables["epi"]:
        if not wanted(table["file"]):
            continue
        for axis, bands in table["bands"].items():
            for band in bands:
                out.append(
                    {
                        "source": table["file"],
                        "axis": axis,
                        "f_hz": band["f_hz"],
                        "tolerance_mt_per_m": band["tolerance_mt_per_m"],
                    }
                )
    return out


def derate_example() -> dict:
    """The 3D GRE built against a system derated to half its gradient
    amplitude: the same areas, every trapezoid longer, the in-band line
    lower; the readout, set by bandwidth and field of view, is untouched."""
    band = (532.0, 600.0)
    kwargs = {
        "n_x": 128,
        "n_y": 64,
        "n_z": 16,
        "tr": None,
        "te": None,
        "readout_bandwidth_hz": 125e3,
    }
    plain = _zoo("gre3D_sequence", SYSTEM, **kwargs)
    derated = _zoo(
        "gre3D_sequence",
        pp.apply_system_derates(SYSTEM, grad_derate=0.5, slew_derate=1.0),
        **kwargs,
    )
    before, after = dense_reading(plain), dense_reading(derated)
    return {
        "band_hz": band,
        "before": before,
        "after": after,
        "before_mt_per_m": max(in_band(before, band, ax) for ax in range(3)),
        "after_mt_per_m": max(in_band(after, band, ax) for ax in range(3)),
    }


#: Gradient coil name to the letter it is called by in a figure, filled in
#: when the tables are read. A published label names no hardware, but two
#: designs that differ only by the coil they were read on must still differ.
COIL_ALIASES: dict[str, str] = {}


def public_label(label: str) -> str:
    """A scenario's label with the gradient coil it was read on taken out."""
    label = re.sub(
        r"in the (\w+) band",
        lambda m: f"in a band on coil {COIL_ALIASES.get(m.group(1), '?')}",
        label,
    )
    label = re.sub(
        r"\(locked on (\w+)\)",
        lambda m: f"(locked TR, coil {COIL_ALIASES.get(m.group(1), '?')})",
        label,
    )
    label = label.replace("multi-echo GRE, ", "multi-echo, ")
    label = re.sub(r", drives \d+ Hz\)", ")", label)
    return label


def _shade(axis, bands, colour=FAINT):
    for band in bands:
        axis.axvspan(*band, color=colour, alpha=0.35, lw=0, zorder=0)


def figure_derate(example: dict) -> Path:
    figure, axis = plt.subplots(figsize=(8.6, 3.2), dpi=170)
    figure.subplots_adjust(top=0.85, bottom=0.16, left=0.1, right=0.98)
    for reading, colour, label in (
        (
            example["before"],
            SERIES[1],
            f"as designed: {example['before_mt_per_m']:.1f} mT/m in band",
        ),
        (
            example["after"],
            SERIES[0],
            f"system derated to half its gradient amplitude: {example['after_mt_per_m']:.1f} mT/m",
        ),
    ):
        f, a = reading["freqs"], reading["amps"].max(axis=1)
        keep = f <= 1500.0
        axis.plot(f[keep], a[keep], lw=0.9, color=colour, label=label)
    _shade(axis, [example["band_hz"]])
    axis.axhline(FLOOR_MT_PER_M, color=INK, lw=0.8, ls=(0, (4, 3)))
    axis.legend(frameon=False, loc="upper right")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("mT/m, worst axis")
    _style(
        axis,
        "3D GRE, TR 6 ms: the spoiler and prewinder carry the line; a derated system moves it",
    )
    out = ASSETS / "derate_example.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def figure_scale() -> Path | None:
    if not SCALE.exists():
        return None
    entries = json.loads(SCALE.read_text())
    if not entries:
        return None
    figure, axis = plt.subplots(figsize=(8.6, 3.4), dpi=170)
    figure.subplots_adjust(top=0.88, bottom=0.16, left=0.1, right=0.98)
    for dims, colour in ((2, SERIES[0]), (3, SERIES[1])):
        rows = sorted(
            (e for e in entries if e["dims"] == dims), key=lambda e: e["arms"]
        )
        if not rows:
            continue
        arms = np.array([e["arms"] for e in rows], float)
        secs = np.array([e["mech_s"] for e in rows], float)
        axis.plot(
            arms, secs, "o-", color=colour, lw=1.0, label=f"{dims}D arms, measured"
        )
        axis.plot(
            [arms[-1], 131072],
            [secs[-1], secs[-1] * 131072 / arms[-1]],
            ls=(0, (3, 3)),
            color=colour,
            lw=0.8,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("distinct arms")
    axis.set_ylabel("check alone (s)")
    axis.legend(frameon=False)
    _style(
        axis,
        "The mechanical-resonance check on a scan of distinct 4096-sample arms; dotted: linear to 128K",
    )
    out = ASSETS / "scale.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


# ----------------------------------------------------------------------
