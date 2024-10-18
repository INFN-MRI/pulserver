# """Test Sequence Definition."""

# import mrd
# import pytest

# from pulserver._core._header import SequenceDefinition


# def test_initialization(mock_mrd):
#     seq_def = SequenceDefinition(3)

#     # Verify the correct number of dimensions is set
#     assert seq_def._ndims == 3

#     # Verify the user parameters are set correctly
#     assert mock_mrd.UserParameterLongType.called
#     mock_mrd.UserParameterLongType.assert_called_with(name="ndims", value=3)


# def test_add_section(mock_mrd):
#     seq_def = SequenceDefinition(3)
#     seq_def.section("first_section")

#     # Verify the section was added
#     assert seq_def._current_section == 0
#     assert "first_section" in seq_def._section_labels


# def test_add_second_section_inherits(mock_mrd):
#     seq_def = SequenceDefinition(3)
#     seq_def.section("first_section")

#     # Define some fields in section 0
#     mock_mrd.EncodingType().encoded_space = "encoded_space_value"
#     mock_mrd.EncodingType().recon_space = "recon_space_value"
#     mock_mrd.EncodingType().encoding_limits = "encoding_limits_value"
#     mock_mrd.EncodingType().trajectory = "trajectory_value"

#     seq_def.section("second_section")

#     # Verify second section inherited the values from section 0
#     assert seq_def._definition.encoding[1].encoded_space == "encoded_space_value"
#     assert seq_def._definition.encoding[1].recon_space == "recon_space_value"
#     assert seq_def._definition.encoding[1].encoding_limits == "encoding_limits_value"
#     assert seq_def._definition.encoding[1].trajectory == "trajectory_value"


# def test_set_shape(mock_mrd):
#     seq_def = SequenceDefinition(3)
#     seq_def.set_definition("shape", 128, 128, 64)

#     # Check that the shape is correctly set for encoded and recon space
#     mock_mrd.EncodingType().encoded_space.matrix_size.x = 128
#     mock_mrd.EncodingType().encoded_space.matrix_size.y = 128
#     mock_mrd.EncodingType().encoded_space.matrix_size.z = 64
#     assert seq_def._shape_set is True


# def test_set_fov_3d(mock_mrd):
#     seq_def = SequenceDefinition(3)
#     seq_def.set_definition("fov", 240.0, 240.0, 240.0)

#     # Check that the FOV is correctly set for encoded and recon space
#     mock_mrd.EncodingType().encoded_space.field_of_view_mm.x = 240.0
#     mock_mrd.EncodingType().encoded_space.field_of_view_mm.y = 240.0
#     mock_mrd.EncodingType().encoded_space.field_of_view_mm.z = 240.0
#     assert mock_mrd.EncodingType().recon_space.field_of_view_mm.x == 240.0
