"""Test automatic segment identification subroutines."""

import itertools

import numpy as np
import numpy.testing as npt

import pytest

from pulserver._core import _autoseg

# events
rf = 8
slice_reph = 9
phase_enc = 3
freq_enc = 1
read_adc = 4
read = 5
delay = 0
spoil = 7
prep = 13
crush = 11

isbal = [True, False]
Ny = [4, 5]
Nz = [4, 5]


def gre(balanced, ny, nz, ndummy):
    # build single readout
    if balanced:
        dummy_segment = np.asarray(
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read,
                freq_enc,
                phase_enc,
                delay,
            ]
        )
        segment = np.asarray(
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                delay,
            ]
        )
    else:
        dummy_segment = np.asarray(
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ]
        )
        segment = np.asarray(
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ]
        )

    # build main loop
    main_loop = np.concatenate([segment for n in range(ny * nz)])

    if ndummy > 0:
        dummy_loop = np.concatenate([dummy_segment for n in range(ndummy)])
        main_loop = np.concatenate([dummy_loop, main_loop])

    return main_loop, dummy_segment


def ir_gre(balanced, ny, nz, dummy):
    # build prep
    prep_segment = np.asarray([prep, crush])

    # build readout
    read_block, dummy_segment = gre(balanced, 1, nz, 0)

    # build main loop
    main_block = np.concatenate([prep_segment, read_block])
    main_loop = np.concatenate([main_block for n in range(ny)])

    if dummy:
        dummy_block = np.concatenate([dummy_segment for n in range(nz)])
        dummy_block = np.concatenate([prep_segment, dummy_block])
        main_loop = np.concatenate([dummy_block, main_loop])

    return main_loop


def ssfp_mrf():
    seq = ir_gre(False, 2, 2, False)
    seq[seq == freq_enc] *= -1  # rotate readout
    seq[seq == read_adc] *= -1
    return seq


def _calc_segment_idx(loop, segments):
    segments_idx = np.zeros(len(loop))
    for n in range(len(segments)):
        tmp = _autoseg.find_segments(loop, segments[n])
        segments_idx[tmp] = n
    segments_idx += 1
    return segments_idx


@pytest.mark.parametrize("isbal, Ny, Nz", list(itertools.product(*[isbal, Ny, Nz])))
def test_find_segment_definitions(isbal, Ny, Nz):
    # case 1: 3D GRE (no dummies)
    loop, _ = gre(isbal, Ny, Nz, 0)
    segments = _autoseg.find_segment_definitions(loop)

    assert len(segments) == 1
    if isbal:
        npt.assert_allclose(
            segments[0],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                delay,
            ],
        )
    else:
        npt.assert_allclose(
            segments[0],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ],
        )

    # case 2: 3D GRE (even dummies)
    loop, _ = gre(isbal, Ny, Nz, 2)
    segments = _autoseg.find_segment_definitions(loop)

    assert len(segments) == 2
    if isbal:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            delay,
        ]
        npt.assert_allclose(segments[0], dummy)
        npt.assert_allclose(
            segments[1],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                delay,
            ],
        )
    else:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        npt.assert_allclose(segments[0], dummy)
        npt.assert_allclose(
            segments[1],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ],
        )

    # case 3: 3D GRE (odd dummies)
    loop, _ = gre(isbal, Ny, Nz, 3)
    segments = _autoseg.find_segment_definitions(loop)
    assert len(segments) == 2
    if isbal:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            delay,
        ]
        npt.assert_allclose(segments[0], dummy)
        npt.assert_allclose(
            segments[1],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                delay,
            ],
        )
    else:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        npt.assert_allclose(segments[0], dummy)
        npt.assert_allclose(
            segments[1],
            [
                rf,
                slice_reph,
                delay,
                phase_enc,
                freq_enc,
                read_adc,
                freq_enc,
                phase_enc,
                spoil,
                delay,
            ],
        )

    # case 4: 3D IR GRE (no dummies)
    loop = ir_gre(isbal, Ny, Nz, False)
    segments = _autoseg.find_segment_definitions(loop)

    assert len(segments) == 1
    if isbal:
        readout = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read_adc,
            freq_enc,
            phase_enc,
            delay,
        ]
        readout = np.concatenate([readout for n in range(Nz)]).tolist()
        npt.assert_allclose(segments[0], [prep, crush] + readout)
    else:
        readout = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read_adc,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        readout = np.concatenate([readout for n in range(Nz)]).tolist()
        npt.assert_allclose(segments[0], [prep, crush] + readout)

    # case 5: 3D IR GRE (dummies)
    loop = ir_gre(isbal, Ny, Nz, True)
    segments = _autoseg.find_segment_definitions(loop)

    assert len(segments) == 2
    if isbal:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            delay,
        ]
        readout = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read_adc,
            freq_enc,
            phase_enc,
            delay,
        ]
        readout = np.concatenate([readout for n in range(Nz)]).tolist()
        npt.assert_allclose(segments[0], [prep, crush] + dummy)
        npt.assert_allclose(segments[1], [prep, crush] + readout)
    else:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        readout = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read_adc,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        readout = np.concatenate([readout for n in range(Nz)]).tolist()
        npt.assert_allclose(segments[0], [prep, crush] + dummy)
        npt.assert_allclose(segments[1], [prep, crush] + readout)


def test_split_rotated_segments():
    loop = ssfp_mrf()
    segments = _autoseg.find_segment_definitions(loop)
    segments = _autoseg.split_rotated_segments(segments)

    # expected segments
    init = [prep, crush, rf, slice_reph, delay, phase_enc]
    readout = [freq_enc, read_adc, freq_enc]
    post_read = [phase_enc, spoil, delay, rf, slice_reph, delay, phase_enc]
    end = [phase_enc, spoil, delay]

    npt.assert_allclose(segments[0], init)
    npt.assert_allclose(segments[1], readout)
    npt.assert_allclose(segments[2], post_read)
    npt.assert_allclose(segments[3], end)


@pytest.mark.parametrize("isbal, Ny, Nz", list(itertools.product(*[isbal, Ny, Nz])))
def test_find_segments(isbal, Ny, Nz):
    # case 1: 3D GRE (no dummies)
    loop, _ = gre(isbal, Ny, Nz, 0)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    expected = np.ones(len(loop))
    npt.assert_allclose(segments_idx, expected)

    # case 2: 3D GRE (even dummies)
    loop, _ = gre(isbal, Ny, Nz, 2)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    if isbal:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            delay,
        ]
    else:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]

    expected = 2 * np.ones(len(loop))
    expected[: 2 * len(dummy)] = 1
    npt.assert_allclose(segments_idx, expected)

    # case 3: 3D GRE (odd dummies)
    loop, _ = gre(isbal, Ny, Nz, 3)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    if isbal:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            delay,
        ]
    else:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]

    expected = 2 * np.ones(len(loop))
    expected[: 3 * len(dummy)] = 1
    npt.assert_allclose(segments_idx, expected)

    # case 4: 3D IR GRE (no dummies)
    loop = ir_gre(isbal, Ny, Nz, False)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    expected = np.ones(len(loop))
    npt.assert_allclose(segments_idx, expected)

    # case 5: 3D IR GRE (dummies)
    loop = ir_gre(isbal, Ny, Nz, True)
    segments = _autoseg.find_segment_definitions(loop)
    segments_idx = _calc_segment_idx(loop, segments)

    if isbal:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            delay,
        ]
        dummy = [prep, crush] + dummy
    else:
        dummy = [
            rf,
            slice_reph,
            delay,
            phase_enc,
            freq_enc,
            read,
            freq_enc,
            phase_enc,
            spoil,
            delay,
        ]
        dummy = [prep, crush] + dummy

    expected = 2 * np.ones(len(loop))
    expected[: 2 + Nz * (len(dummy) - 2)] = 1
    npt.assert_allclose(segments_idx, expected)
