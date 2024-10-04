"""Test SequenceParams structure."""

import struct

from pulserver._core._ceq import SequenceParams


def test_sequence_params_from_bytes():
    """
    Test deserialization from bytes.
    """
    # Create a test byte array equivalent to the C struct filled with -1's
    test_data = struct.pack(
        "2f 5h 2f 5f h 5f 3f 3h h 6f",
        -1.0,
        -1.0,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1,
        -1,
        -1,
        -1,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    )

    # Deserialize into a SequenceParams object
    params = SequenceParams.from_bytes(test_data)

    # Ensure all fields are set to None
    assert params.FOVx is None
    assert params.FOVy is None
    assert params.Nx is None
    assert params.Ny is None
    assert params.Nslices is None
    assert params.Nechoes is None
    assert params.Nphases is None
    assert params.slice_thickness is None
    assert params.slice_spacing is None
    assert params.Rplane is None
    assert params.Rplane2 is None
    assert params.Rslice is None
    assert params.PFplane is None
    assert params.PFslice is None
    assert params.ETL is None
    assert params.TE is None
    assert params.TE0 is None
    assert params.TR is None
    assert params.Tprep is None
    assert params.Trecovery is None
    assert params.flip is None
    assert params.flip2 is None
    assert params.refoc_flip is None
    assert params.freq_dir is None
    assert params.freq_verse is None
    assert params.phase_verse is None
    assert params.bipolar_echoes is None
    assert params.dwell is None
    assert params.raster is None
    assert params.gmax is None
    assert params.smax is None
    assert params.b1_max is None
    assert params.b0_field is None


def test_sequence_params_asdict():
    """
    Test the asdict method to exclude None values.
    """
    params = SequenceParams(
        FOVx=150.0, Nx=128, Ny=None, flip=30.0, dwell=0.01, gmax=40.0
    )

    params_dict = params.asdict()

    # Ensure the dict excludes None values
    assert params_dict == {
        "FOVx": 150.0,
        "Nx": 128,
        "flip": 30.0,
        "dwell": 0.01,
        "gmax": 40.0,
    }

    # Ensure 'Ny' and unset fields are not included in the dict because they are None
    assert "Ny" not in params_dict
    assert "b1_max" not in params_dict
