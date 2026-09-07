#!/usr/bin/env python3
"""The corpus the mechanical-resonance constants are calibrated on.

Every design is read against every gradient coil's forbidden bands, and the
product's verdict for that pair comes from one rule applied to every checked
family: the coil's echo-spacing tables give the forbidden frequency bands per
axis, each with the readout plateau it tolerates; the family's locked
parameter is converted to the frequency it drives on the axis that carries it
(``drives``) and refused when it lands in a band above that plateau
(``product_verdict``). A family whose parameter drives nothing the tables look
at is "not checked".

The two constants the check needs -- the resonance memory and the floor a band
that states no amplitude of its own is held to -- are then chosen to agree
with the product as often as the corpus allows.
"""

from __future__ import annotations

import math

import numpy as np

#: Enough repetitions to fill the longest window the sweep reads at, with
#: margin: a scan cut to this length reads what the whole scan reads, and the
#: reading is what costs.
LONGEST_WINDOW_S = 0.080
#: Windows the corpus is re-read at, in seconds.
WINDOWS_S = (
    0.0025,
    0.005,
    0.0075,
    0.010,
    0.0125,
    0.015,
    0.0175,
    0.020,
    0.025,
    0.030,
    0.040,
    0.060,
    0.080,
)


def repetitions_for(tr_s: float, minimum: int = 4) -> int:
    """How many repetitions a scan needs for the longest window to be full."""
    if not np.isfinite(tr_s) or tr_s <= 0.0:
        return minimum
    return max(minimum, math.ceil(LONGEST_WINDOW_S / tr_s) + 2)


def protocols(zoo, system, weak) -> list[dict]:
    """Every design the corpus measures.

    One or two knobs per family, each chosen because it moves the drive
    frequency, its amplitude or its duration: the echo spacing and readout
    plateau of a train, the repetition time and slice thickness of a balanced
    steady state, the polarity and echo count of a multi-echo train, the
    spoiler of a spoiled gradient echo, the arm count of a spiral.
    """
    out: list[dict] = []

    def add(family, label, build):
        out.append({"family": family, "label": label, "build": build})

    # Echo-planar: the matrix and the receiver bandwidth set the echo spacing,
    # and the bandwidth sets the readout plateau with it.
    for n in (32, 48, 64, 96, 128):
        for bw in (60e3, 100e3, 160e3):
            add(
                "epi",
                f"EPI {n}, {bw / 1e3:.0f} kHz",
                lambda n=n, bw=bw: zoo(
                    "epi2D_sequence", system, n_x=n, n_y=n, readout_bandwidth_hz=bw
                ),
            )
    add(
        "epi",
        "EPI 96, 100 kHz, derated gradients",
        lambda: zoo("epi2D_sequence", weak, n_x=96, n_y=96, readout_bandwidth_hz=100e3),
    )

    # Balanced steady state: the repetition time sets the comb, and in 2D the
    # slice select carries more of it than the readout does.
    for tr in (3.3e-3, 3.4e-3, 3.5e-3, 3.9e-3, 4.5e-3, 5.15e-3, 6.0e-3):
        add(
            "bssfp",
            f"bSSFP 2D, TR {tr * 1e3:.2f} ms",
            lambda tr=tr: zoo("bssfp2D_sequence", system, n_x=128, n_y=32, tr=tr),
        )
    for thickness in (8e-3,):
        for tr in (3.4e-3, 4.5e-3):
            add(
                "bssfp",
                f"bSSFP 2D, TR {tr * 1e3:.2f} ms, {thickness * 1e3:.0f} mm slice",
                lambda tr=tr, th=thickness: zoo(
                    "bssfp2D_sequence",
                    system,
                    n_x=128,
                    n_y=32,
                    tr=tr,
                    slice_thickness=th,
                ),
            )
    for tr in (3.4e-3, 4.5e-3, 6.0e-3):
        add(
            "bssfp",
            f"bSSFP 3D, TR {tr * 1e3:.2f} ms",
            lambda tr=tr: zoo(
                "bssfp3D_sequence", system, n_x=128, n_y=16, n_z=8, tr=tr
            ),
        )
    return out


def more_protocols(zoo, system, weak, multiecho_at) -> list[dict]:
    """The rest of the corpus: the families the tables cover through a spacing,
    and the families no table covers at all."""
    out: list[dict] = []

    def add(family, label, build):
        out.append({"family": family, "label": label, "build": build})

    # Multi-echo: the spacing sets the drive, and the polarity decides whether
    # the drive period is one spacing or two. Echo count sets the train length.
    for esp_us in (1400.0, 1700.0, 2100.0):
        for monopolar in (True, False):
            for n_echoes in (4, 12):
                kind = "flyback" if monopolar else "bipolar"
                add(
                    f"multiecho_{kind}",
                    f"multi-echo, {n_echoes} {kind}, ESP {esp_us / 1e3:.2f} ms",
                    lambda e=esp_us, m=monopolar, n=n_echoes: multiecho_at(
                        e * 1e-6, n, m
                    ),
                )

    # Spoiled gradient echo, and its inversion-prepared form at the same
    # readout repetition times: the train between inversions is a spoiled
    # gradient echo, so the two should agree once the train is long enough
    # for the inversion and the trailing wait to be a small part of the duty.
    for tr in (12e-3, 20e-3, 100e-3):
        add(
            None,
            f"2D GRE, TR {tr * 1e3:.0f} ms",
            lambda tr=tr: zoo(
                "gre2D_sequence",
                system,
                n_x=128,
                n_y=12,
                tr=tr,
                readout_bandwidth_hz=100e3,
            ),
        )
    for tr in (6e-3, 8e-3, 10e-3):
        add(
            None,
            f"3D GRE, TR {tr * 1e3:.0f} ms",
            lambda tr=tr: zoo(
                "gre3D_sequence",
                system,
                n_x=96,
                n_y=32,
                n_z=4,
                tr=tr,
                readout_bandwidth_hz=100e3,
            ),
        )
        for views in (16, 64, 128):
            add(
                None,
                f"MPRAGE, TR {tr * 1e3:.0f} ms, {views} views per inversion",
                lambda tr=tr, v=views: zoo(
                    "mprage3D_sequence",
                    system,
                    n_x=96,
                    n_y=32,
                    n_z=4,
                    views_per_segment=v,
                    tr=tr,
                    readout_bandwidth_hz=100e3,
                ),
            )
    add(
        None,
        "MPRAGE, derated gradients",
        lambda: zoo("mprage3D_sequence", weak, n_x=96, n_y=32, n_z=4),
    )

    # Spin-echo trains: the echo spacing is set by the refocusing pulse and
    # its crushers, so it sits above the balanced steady-state repetition
    # times; the train length and readout bandwidth are what vary.
    for etl, bw in ((8, 60e3), (16, 250e3), (32, 250e3)):
        add(
            None,
            f"FSE 2D, ETL {etl}, {bw / 1e3:.0f} kHz",
            lambda etl=etl, bw=bw: zoo(
                "fse2D_sequence",
                system,
                n_x=128,
                n_y=etl * 2,
                etl=etl,
                te=None,
                tr=None,
                readout_bandwidth_hz=bw,
            ),
        )
    for etl, bw in ((32, 125e3), (32, 250e3), (64, 125e3)):
        add(
            None,
            f"FSE 3D, ETL {etl}, {bw / 1e3:.0f} kHz",
            lambda etl=etl, bw=bw: zoo(
                "fse3D_sequence",
                system,
                n_x=128,
                n_y=16,
                n_z=8,
                etl=etl,
                te=None,
                tr=None,
                readout_bandwidth_hz=bw,
            ),
        )

    # Trajectories no lockout mentions: a sweep's rate is set by how many arms
    # share the trajectory, a spin-echo train's comb by its echo spacing.
    for arms in (1, 4, 8, 16):
        add(
            None,
            f"spiral, {arms} arm{'s' if arms > 1 else ''}",
            lambda a=arms: zoo(
                "gre_spiral2D_sequence",
                weak if a == 1 else system,
                n_x=128,
                n_arms=a,
                tr=None,
            ),
        )
    for tr in (5e-3, 10e-3, 20e-3):
        add(
            None,
            f"radial GRE, TR {tr * 1e3:.0f} ms",
            lambda tr=tr: zoo(
                "gre_radial2D_sequence", system, n_x=128, n_spokes=32, tr=tr
            ),
        )
    add(
        None,
        "stack of stars",
        lambda: zoo(
            "gre_stack_of_stars3D_sequence", system, n_x=128, n_spokes=32, n_z=4
        ),
    )
    add(
        None,
        "stack of spirals",
        lambda: zoo(
            "gre_stack_of_spirals3D_sequence", system, n_x=128, n_arms=8, n_z=4
        ),
    )
    add(None, "spin echo", lambda: zoo("se2D_sequence", system, n_x=128, n_y=12))
    add(
        None,
        "PROPELLER",
        lambda: zoo(
            "se_propeller2D_sequence", system, n_x=64, readout_bandwidth_hz=100e3
        ),
    )
    add(
        None,
        "ZTE",
        lambda: zoo("zte3D_sequence", system, n_x=48, readout_bandwidth_hz=100e3),
    )
    return out


def stated_tolerances(sequence, bands, gamma_hz_per_mt_per_m) -> dict:
    """What the engine judges each stated-amplitude band against, in mT/m.

    A band that states a plateau is converted to a sinusoid through the shape
    of the train that drives it, which is a property of the design; asking the
    engine keeps the corpus and the gate on one definition instead of two.
    """
    stated = [b for b in bands if b["tolerance_mt_per_m"] > 0.0]
    if not stated:
        return {}
    request = [
        (
            b["f_hz"][0],
            b["f_hz"][1],
            b["tolerance_mt_per_m"] * gamma_hz_per_mt_per_m,
            f"g{b['axis']}" if b["axis"] else "",
        )
        for b in stated
    ]
    overlay = sequence.calculate_gradient_spectrum(
        plot=False,
        max_frequency=max(b["f_hz"][1] for b in stated) * 1.2,
        tr="worst_case",
        resonance_lines=True,
        bands=request,
    )[4]
    freqs = np.asarray(overlay.candidate_freqs, float)
    tol = np.asarray(overlay.tolerance, float) / gamma_hz_per_mt_per_m
    out = {}
    for b in stated:
        inside = (freqs >= b["f_hz"][0]) & (freqs <= b["f_hz"][1])
        if inside.any():
            out[(round(b["f_hz"][0], 3), round(b["f_hz"][1], 3), b["axis"])] = float(
                tol[inside].max()
            )
    return out


#: How close to a band edge a design's drive may sit and still be evidence,
#: as a fraction of the edge. A lockout is a step -- refused on one side,
#: permitted on the other -- and a drive within grid precision of the step is
#: decided by rounding, not by the product. Such pairs are set aside. The
#: check's own resolution limit is a different thing and is counted.
BOUNDARY_FRACTION = 0.01


def slice_select_plateau_mt_per_m(sequence, gamma_hz_per_mt_per_m) -> float:
    """The largest slice-select amplitude played under an RF pulse."""
    peak = 0.0
    for index in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(index)
        if block.rf is None or block.gz is None:
            continue
        grad = block.gz
        value = (
            abs(float(grad.amplitude))
            if grad.type == "trap"
            else float(np.max(np.abs(np.asarray(grad.waveform, float))))
        )
        peak = max(peak, value)
    return peak / gamma_hz_per_mt_per_m


def drives(family, esp_s, tr_s, plateau_x, plateau_z):
    """[(axis, frequency, plateau)] the product's rule looks at.

    An echo-planar or bipolar train alternates and drives 1 / (2 ESP) on its
    readout axis; a flyback train repeats every spacing and drives 1 / ESP; a
    balanced steady state drives 2 / TR on its readout axis and on its slice
    axis. Families no table covers drive nothing the product looks at.
    """
    if family in ("epi", "multiecho_bipolar") and np.isfinite(esp_s) and esp_s > 0:
        return [("x", 0.5 / esp_s, plateau_x)]
    if family == "multiecho_flyback" and np.isfinite(esp_s) and esp_s > 0:
        return [("x", 1.0 / esp_s, plateau_x)]
    if family == "bssfp" and np.isfinite(tr_s) and tr_s > 0:
        return [("x", 2.0 / tr_s, plateau_x), ("z", 2.0 / tr_s, plateau_z)]
    return []


def product_verdict(drive_list, bands) -> str:
    """What the coil's echo-spacing table says about a design -- one rule for
    every checked family. Refused when a drive falls in a band on the axis
    that carries it and its plateau exceeds the band's tolerance, a zero
    tolerance refusing any plateau; not checked when the family drives nothing
    the table looks at; on a boundary when a drive sits on a band edge."""
    if not drive_list:
        return "not checked"
    out = "accepted"
    for band in bands:
        lo, hi = band["f_hz"]
        for axis, f, plateau in drive_list:
            if band["axis"] not in (None, axis):
                continue
            if (
                abs(f - lo) < BOUNDARY_FRACTION * lo
                or abs(f - hi) < BOUNDARY_FRACTION * hi
            ):
                return "on a boundary"
            if lo <= f <= hi and (
                band["tolerance_mt_per_m"] <= 0.0
                or plateau > band["tolerance_mt_per_m"]
            ):
                out = "refused"
    return out


def refuses(stated_ratio: float, zero_reading: float, floor: float) -> bool:
    """Whether the check refuses a design on a coil at a given floor.

    A band that states an amplitude is judged against it whatever the floor;
    only the bands that state nothing move with the constant being calibrated.
    """
    return stated_ratio > 1.0 or zero_reading > floor


VERDICT_COLOUR = {"refused": 1, "accepted": 0, "not checked": None}
VERDICT_LABEL = {
    "refused": "checked, refused",
    "accepted": "checked, accepted",
    "not checked": "not checked",
}


def ratio_of(row, floor):
    """A pair's reading against what its band tolerates: one means at it."""
    return max(row["stated_ratio"], row["zero_reading"] / floor)


def disagreements(rows, floor: float) -> dict:
    """Every pair on which this check and the product part, by name.

    Three lists: what the product refuses and this check passes, what it
    accepts and this check refuses, and what it never checks and this check
    refuses. An entry names the design, the coils it happens on and the
    loudest reading among them as a ratio to the threshold.
    """
    buckets = {
        "product refuses, this check passes": ("refused", False),
        "product accepts, this check refuses": ("accepted", True),
        "product does not check, this check refuses": ("not checked", True),
    }
    out = {}
    for name, (verdict, hit) in buckets.items():
        found: dict[str, dict] = {}
        for r in rows:
            ratio = ratio_of(r, floor)
            if r["verdict"] != verdict or (ratio > 1.0) != hit:
                continue
            entry = found.setdefault(
                r["label"], {"label": r["label"], "coils": [], "ratio_up_to": 0.0}
            )
            entry["coils"].append(r["coil"])
            entry["ratio_up_to"] = max(entry["ratio_up_to"], ratio)
        out[name] = sorted(
            found.values(), key=lambda e: (-len(e["coils"]), -e["ratio_up_to"])
        )
    return out


def figure_window(records, floor, assets, colours, ink, style):
    """Every checked pair against the window, and what the floor does there."""
    import matplotlib.pyplot as plt

    windows = sorted(records)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 6.0), dpi=170, sharex=True, height_ratios=(1.6, 1.0)
    )
    figure.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.10, hspace=0.28)
    x = [w * 1e3 for w in windows]
    for axis in (top, bottom):
        axis.axvspan(
            20.0 * 0.93, 20.0 * 1.07, color=colours[1], alpha=0.10, lw=0, zorder=0
        )
    reproduced, false = [], []
    for w in windows:
        rows = records[w]
        for verdict, shade, shift in (
            ("refused", colours[1], 1.03),
            ("accepted", colours[0], 0.97),
        ):
            values = [ratio_of(r, floor) for r in rows if r["verdict"] == verdict]
            top.plot(
                [w * 1e3 * shift] * len(values),
                values,
                "o",
                ms=3.5,
                color=shade,
                alpha=0.6,
                zorder=3,
            )
        refused = [r for r in rows if r["verdict"] == "refused"]
        accepted = [r for r in rows if r["verdict"] == "accepted"]
        reproduced.append(
            100.0 * sum(ratio_of(r, floor) > 1 for r in refused) / max(1, len(refused))
        )
        false.append(
            100.0
            * sum(ratio_of(r, floor) > 1 for r in accepted)
            / max(1, len(accepted))
        )
    top.plot([], [], "o", ms=3.5, color=colours[1], label=VERDICT_LABEL["refused"])
    top.plot([], [], "o", ms=3.5, color=colours[0], label=VERDICT_LABEL["accepted"])
    top.axhline(1.0, ls="--", lw=1.0, color=ink)
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylim(0.05, 6.0)
    top.set_ylabel("mechanical response / threshold")
    top.legend(frameon=False, fontsize=8, loc="upper right")
    style(top, "(a) every checked pair, against the window")
    bottom.plot(
        x,
        reproduced,
        "-o",
        ms=4,
        color=colours[1],
        label="of those the product refuses, refused here",
    )
    bottom.plot(
        x,
        false,
        "-o",
        ms=4,
        color=colours[0],
        label="of those it accepts, refused here",
    )
    bottom.set_xscale("log")
    bottom.set_xticks(x)
    bottom.set_xticklabels([f"{v:g}" for v in x], fontsize=8)
    bottom.minorticks_off()
    bottom.set_ylim(-8, 118)
    bottom.set_yticks([0, 25, 50, 75, 100])
    bottom.set_xlabel("resonance memory W (ms)")
    bottom.set_ylabel("% of pairs")
    bottom.legend(frameon=False, fontsize=8, loc="upper right")
    style(bottom, "(b) what the shipped threshold does at each window")
    out = assets / "window.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def figure_corpus(rows, floor, assets, colours, muted, ink, style, public_label):
    """Every design at the chosen window: its loudest reading across the coils
    on which the product has a verdict, coloured by that verdict."""
    import matplotlib.pyplot as plt

    loudest = {}
    for r in rows:
        if r["verdict"] == "on a boundary":
            continue
        ratio = ratio_of(r, floor)
        if r["label"] not in loudest or ratio > loudest[r["label"]][0]:
            loudest[r["label"]] = (ratio, r["verdict"])
    groups = (("refused", colours[1]), ("accepted", colours[0]), ("not checked", muted))
    positions, values, shades, labels, headers = [], [], [], [], []
    at = 0.0
    for verdict, shade in groups:
        members = sorted(
            ((v[0], k) for k, v in loudest.items() if v[1] == verdict),
        )
        if not members:
            continue
        for ratio, label in members:
            positions.append(at)
            values.append(ratio)
            shades.append(shade)
            labels.append(public_label(label))
            at += 1.0
        refused_here = sum(1 for ratio, _ in members if ratio > 1.0)
        headers.append(
            (
                at,
                f"{VERDICT_LABEL[verdict]} — {refused_here} of {len(members)} refused by this check",
                shade,
            )
        )
        at += 1.4
    figure, axis = plt.subplots(figsize=(9.4, 0.28 * at + 1.4), dpi=170)
    figure.subplots_adjust(left=0.42, right=0.97, top=0.97, bottom=0.08)
    axis.barh(positions, values, color=shades, height=0.68, zorder=2)
    over = [(y, v) for y, v in zip(positions, values, strict=False) if v > 1.0]
    if over:
        axis.plot(
            [v for _, v in over],
            [y for y, _ in over],
            "o",
            ms=4.5,
            color=ink,
            zorder=4,
            clip_on=False,
        )
    axis.axvline(1.0, color=ink, lw=1.1, ls=(0, (4, 3)), zorder=3)
    axis.set_xscale("log")
    axis.set_xlim(0.03, 4.0)
    axis.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    axis.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"], fontsize=8.5)
    axis.minorticks_off()
    axis.set_ylim(-1.0, at - 0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=7.5)
    axis.set_xlabel("mechanical response / threshold")
    for y, title, shade in headers:
        axis.text(0.032, y + 0.15, title, fontsize=9, color=shade, va="center")
    axis.annotate(
        "threshold; dots mark what this check refuses",
        (1.0, -0.9),
        textcoords="offset points",
        xytext=(-5, 0),
        ha="right",
        fontsize=8.5,
        color=ink,
    )
    style(axis, "")
    out = assets / "corpus.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def main() -> int:
    import argparse
    import importlib.util
    import json
    import sys
    import time
    import warnings
    from pathlib import Path

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("mc", here / "mechres_calibration.py")
    mc = importlib.util.module_from_spec(spec)
    sys.modules["mc"] = mc
    spec.loader.exec_module(mc)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="a few designs only")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    tables = mc.load_tables()
    coils = sorted({mc.coil_of(t["file"]) for t in tables["epi"]})
    per_coil = {c: mc.vendor_bands(tables, {c}) for c in coils}
    upper = min(
        mc.TRAIN_SHAPE * b["tolerance_mt_per_m"]
        for bands in per_coil.values()
        for b in bands
        if b["tolerance_mt_per_m"] > 0.0
    )

    def multiecho_at(esp_s, n_echoes, monopolar):
        """The candidate at that spacing whose readout plateau is nearest the
        one the tables' stated tolerance is written for -- a prescription a
        console offers, not the loudest the amplifier allows."""
        found = mc.multiecho_with_spacing_in(
            (esp_s * 0.94, esp_s * 1.06), n_echoes, monopolar
        )
        if not found:
            return None
        return min(
            found,
            key=lambda c: abs(
                mc._readout_plateau_mt_per_m(c[0]) - mc.TARGET_PLATEAU_MT_PER_M
            ),
        )[0]

    designs = protocols(mc._zoo, mc.SYSTEM, mc.WEAK) + more_protocols(
        mc._zoo, mc.SYSTEM, mc.WEAK, multiecho_at
    )
    if args.quick:
        designs = designs[::7]
    print(
        f"{len(designs)} designs, {len(coils)} coils, floor bounded by {upper:.2f} mT/m"
    )

    built = []
    for entry in designs:
        try:
            seq = entry["build"]()
        except Exception as exc:
            print(f"  {entry['label']}: not built ({str(exc)[:60]})", flush=True)
            continue
        if seq is None:
            print(f"  {entry['label']}: no design reaches it", flush=True)
            continue
        entry["sequence"] = seq
        entry["esp_s"] = mc._echo_spacing_s(seq)
        entry["tr_s"] = mc._repetition_time_s(seq)
        entry["plateau_x"] = mc._readout_plateau_mt_per_m(seq)
        entry["plateau_z"] = slice_select_plateau_mt_per_m(
            seq, mc.GAMMA_HZ_PER_MT_PER_M
        )
        entry["blocks"] = len(seq.block_events)
        built.append(entry)
        print(f"  {entry['label']}: {entry['blocks']} blocks", flush=True)

    all_bands = [b for bands in per_coil.values() for b in bands]
    for entry in built:
        entry["stated"] = stated_tolerances(
            entry["sequence"], all_bands, mc.GAMMA_HZ_PER_MT_PER_M
        )
        entry["drives"] = drives(
            entry["family"],
            entry["esp_s"],
            entry["tr_s"],
            entry["plateau_x"],
            entry["plateau_z"],
        )
        entry["verdicts"] = {
            c: product_verdict(entry["drives"], per_coil[c]) for c in coils
        }

    axes = "xyz"
    records: dict[float, list] = {}
    per_band: dict[str, dict] = {}
    t0 = time.perf_counter()
    for window in WINDOWS_S:
        rows = []
        for entry in built:
            reading = mc.dense_reading(entry["sequence"], memory=window)
            slot = per_band.setdefault(entry["label"], {})
            for key, b in {
                (round(b["f_hz"][0], 3), round(b["f_hz"][1], 3), b["axis"]): b
                for b in all_bands
            }.items():
                value = max(
                    mc.in_band(reading, b["f_hz"], ax)
                    for ax in range(3)
                    if b["axis"] is None or axes[ax] == b["axis"]
                )
                slot.setdefault(f"{key[0]}|{key[1]}|{key[2]}", {})[str(window)] = value
            for coil in coils:
                stated_ratio = zero_reading = 0.0
                for band in per_coil[coil]:
                    key = (
                        round(band["f_hz"][0], 3),
                        round(band["f_hz"][1], 3),
                        band["axis"],
                    )
                    for ax in range(3):
                        if band["axis"] is not None and axes[ax] != band["axis"]:
                            continue
                        value = mc.in_band(reading, band["f_hz"], ax)
                        if band["tolerance_mt_per_m"] > 0.0:
                            tol = entry["stated"].get(
                                key, mc.TRAIN_SHAPE * band["tolerance_mt_per_m"]
                            )
                            stated_ratio = max(stated_ratio, value / tol)
                        else:
                            zero_reading = max(zero_reading, value)
                verdict = entry["verdicts"][coil]
                rows.append(
                    {
                        "label": entry["label"],
                        "family": entry["family"],
                        "coil": coil,
                        "verdict": verdict,
                        "stated_ratio": stated_ratio,
                        "zero_reading": zero_reading,
                    }
                )
        records[window] = rows
        print(
            f"  W {window * 1e3:5.1f} ms read ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )

    summary = []
    print(f"\nat the shipped floor, {mc.FLOOR_MT_PER_M:.0f} mT/m:")
    for window, rows in records.items():
        refused = [r for r in rows if r["verdict"] == "refused"]
        accepted = [r for r in rows if r["verdict"] == "accepted"]
        unchecked = [r for r in rows if r["verdict"] == "not checked"]
        hit = lambda r: refuses(r["stated_ratio"], r["zero_reading"], mc.FLOOR_MT_PER_M)
        entry = {
            "window_s": window,
            "refused_pairs": len(refused),
            "reproduced": sum(1 for r in refused if hit(r)),
            "accepted_pairs": len(accepted),
            "false_refusals": sum(1 for r in accepted if hit(r)),
            "unchecked_pairs": len(unchecked),
            "unchecked_refused": sum(1 for r in unchecked if hit(r)),
            "boundary_pairs": sum(1 for r in rows if r["verdict"] == "on a boundary"),
        }
        summary.append(entry)
        print(
            f"  W {window * 1e3:5.1f} ms  reproduces {entry['reproduced']:2d}/{entry['refused_pairs']}"
            f"  falsely refuses {entry['false_refusals']:3d}/{entry['accepted_pairs']}"
            f"  unchecked refused {entry['unchecked_refused']:2d}/{entry['unchecked_pairs']}"
        )

    named = disagreements(records[0.020], mc.FLOOR_MT_PER_M)
    for name, entries in named.items():
        print(f"\n{name}: {sum(len(e['coils']) for e in entries)} pairs")
        for e in entries:
            print(
                f"  {e['label']} on {', '.join(e['coils'])}"
                f"  (up to {e['ratio_up_to']:.2f}x)"
            )

    assets = here.parents[0] / "explanations" / "assets" / "mechanical_resonance"
    assets.mkdir(parents=True, exist_ok=True)
    from _figures import INK, MUTED, SERIES, _style

    print(
        "wrote",
        figure_window(records, mc.FLOOR_MT_PER_M, assets, SERIES, INK, _style).name,
    )
    print(
        "wrote",
        figure_corpus(
            records[0.020],
            mc.FLOOR_MT_PER_M,
            assets,
            SERIES,
            MUTED,
            INK,
            _style,
            mc.public_label,
        ).name,
    )
    out = here / "mechres_corpus.json"
    out.write_text(
        json.dumps(
            {
                "coils": coils,
                "floor_upper_mt_per_m": upper,
                "summary": summary,
                "disagreements": named,
                "records": {str(w): r for w, r in records.items()},
                "designs": {
                    e["label"]: {
                        "family": e["family"],
                        "esp_s": float(e["esp_s"]),
                        "tr_s": float(e["tr_s"]),
                        "plateau_x": float(e["plateau_x"]),
                        "plateau_z": float(e["plateau_z"]),
                        "stated": {
                            f"{k[0]}|{k[1]}|{k[2]}": v for k, v in e["stated"].items()
                        },
                        "verdicts": e["verdicts"],
                    }
                    for e in built
                },
                "bands": {
                    f"{round(b['f_hz'][0], 3)}|{round(b['f_hz'][1], 3)}|{b['axis']}": {
                        "f_hz": b["f_hz"],
                        "axis": b["axis"],
                        "tolerance_mt_per_m": b["tolerance_mt_per_m"],
                        "coil": c,
                    }
                    for c in coils
                    for b in per_coil[c]
                },
                "per_band": per_band,
            },
            indent=1,
            default=float,
        )
        + "\n"
    )
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
