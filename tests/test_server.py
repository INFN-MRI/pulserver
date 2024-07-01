"""Test suite for server."""

import logging
import os
import pytest
import socket
import tempfile

from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

from pulseforge.server._server import load_plugins
from pulseforge.server._server import parse_request
from pulseforge.server._server import setup_function_logger
from pulseforge.server._server import send_to_recon_server
from pulseforge.server._server import handle_client_connection
from pulseforge.server import start_server

# Create a logger for the function (this is just to match the function's logging behavior)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def create_mock_plugins():
    with tempfile.TemporaryDirectory() as tempdir:
        # Create mock plugin files
        plugin_code = """
def test_func():
    return "Hello, world!"
"""
        plugin1_path = os.path.join(tempdir, "plugin1.py")
        with open(plugin1_path, "w") as f:
            f.write(plugin_code)

        plugin2_path = os.path.join(tempdir, "plugin2.py")
        with open(plugin2_path, "w") as f:
            f.write(plugin_code)

        yield tempdir


def test_load_plugins(create_mock_plugins, caplog):
    plugin_dir = create_mock_plugins
    os.environ["PULSEFORGE_PLUGINS"] = plugin_dir

    with caplog.at_level(logging.DEBUG):
        plugins = load_plugins(logger)

        # Check that the plugins are loaded correctly
        assert "plugin1" in plugins
        assert "plugin2" in plugins

        # Check that the functions in the plugins work correctly
        assert plugins["plugin1"].test_func() == "Hello, world!"
        assert plugins["plugin2"].test_func() == "Hello, world!"

        # Verify that the debug log was called correctly
        expected_log_message1 = (
            f"Loaded plugin: plugin1 from {os.path.join(plugin_dir, 'plugin1.py')}"
        )
        expected_log_message2 = (
            f"Loaded plugin: plugin2 from {os.path.join(plugin_dir, 'plugin2.py')}"
        )

        assert any(
            expected_log_message1 in message for message in caplog.text.splitlines()
        )
        assert any(
            expected_log_message2 in message for message in caplog.text.splitlines()
        )


def test_parse_request_valid(caplog):
    request = "foo 2 var1 var2"
    with caplog.at_level(logging.DEBUG):
        function_name, args = parse_request(request, logger)

    assert function_name == "foo"
    assert args == ["var1", "var2"]
    assert "Parsed request - Function: foo, Args: ['var1', 'var2']" in caplog.text


def test_parse_request_invalid_format(caplog):
    request = "foo two var1 var2"
    with caplog.at_level(logging.ERROR):
        function_name, args = parse_request(request, logger)

    assert function_name is None
    assert args is None
    assert "Failed to parse request: invalid literal for int()" in caplog.text


def test_parse_request_missing_args(caplog):
    request = "foo 2 var1"
    with caplog.at_level(logging.DEBUG):
        function_name, args = parse_request(request, logger)

    assert function_name == "foo"
    assert args == ["var1"]
    assert "Parsed request - Function: foo, Args: ['var1']" in caplog.text


def test_parse_request_extra_args(caplog):
    request = "foo 2 var1 var2 var3"
    with caplog.at_level(logging.DEBUG):
        function_name, args = parse_request(request, logger)

    assert function_name == "foo"
    assert args == ["var1", "var2"]
    assert "Parsed request - Function: foo, Args: ['var1', 'var2']" in caplog.text


# The function to be tested
LOG_DIR = "/path/to/log/dir"  # Update with appropriate log directory for testing


@pytest.fixture
def temp_log_dir(tmp_path):
    # Create a temporary directory for logs
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    global LOG_DIR
    LOG_DIR = str(log_dir)
    yield log_dir


@patch("datetime.datetime")
def test_setup_function_logger(mock_datetime, temp_log_dir):
    function_name = "test_function"

    # Set up the mock datetime to return a fixed time
    fixed_datetime = datetime(2024, 7, 1, 12, 0, 0)
    mock_datetime.now.return_value = fixed_datetime
    mock_datetime.strftime = fixed_datetime.strftime  # Use the real strftime method

    function_logger = setup_function_logger(function_name)

    # Use a general pattern for the filename instead of an exact match
    log_filename_pattern = f"{function_name}_"

    # Check that the logger was set up correctly
    assert function_logger.name == function_name
    assert function_logger.level == logging.DEBUG
    assert len(function_logger.handlers) == 1

    handler = function_logger.handlers[0]
    assert isinstance(handler, logging.FileHandler)

    # Check that the log file name contains the expected pattern
    base_filename = os.path.basename(handler.baseFilename)
    assert base_filename.startswith(log_filename_pattern)
    assert base_filename.endswith(".log")

    # Check that the log file is created
    assert os.path.exists(handler.baseFilename)

    # Optionally, check that the timestamp part of the filename is in the expected format
    # Split the filename by the function name to get the timestamp part
    timestamp_part = base_filename[len(function_name) + 1 : -4]
    assert len(timestamp_part) == 15  # Expect "YYYYMMDD_HHMMSS" format
    assert timestamp_part[:8].isdigit()  # YYYYMMDD
    assert timestamp_part[9:].isdigit()  # HHMMSS


# Constants for the test
MR_SCANNER_ADDRESS = "localhost"
MR_SCANNER_PORT = 12345
RECON_SERVER_ADDRESS = "localhost"
RECON_SERVER_PORT = 23456


# Define a sample configuration for the tests
@pytest.fixture
def sample_config():
    return {
        "scanner_address": MR_SCANNER_ADDRESS,
        "scanner_port": MR_SCANNER_PORT,
        "recon_server_address": RECON_SERVER_ADDRESS,
        "recon_server_port": RECON_SERVER_PORT,
    }


# Test for send_to_recon_server with valid config
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_valid_config(mock_socket, sample_config):
    # Create a mock socket instance
    mock_socket_instance = MagicMock()
    mock_socket.return_value.__enter__.return_value = mock_socket_instance

    # Example buffer data
    optional_buffer = b"test data"

    # Call the function
    send_to_recon_server(b"test data", sample_config)

    # Check that the socket was created with the correct arguments
    mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)

    # Check that the connect method was called with the correct arguments
    mock_socket_instance.connect.assert_called_once_with(
        (RECON_SERVER_ADDRESS, RECON_SERVER_PORT)
    )

    # Check that the sendall method was called with the correct buffer
    mock_socket_instance.sendall.assert_called_once_with(optional_buffer)


# Test for send_to_recon_server with missing address in config
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_missing_address(mock_socket):
    config = {"recon_server_port": RECON_SERVER_PORT}
    # Call the function with missing address in config
    # It should not raise an exception or try to connect
    send_to_recon_server(b"test data", config)

    # Ensure that socket methods are not called due to missing address
    mock_socket.assert_not_called()


# Test for send_to_recon_server with missing port in config
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_missing_port(mock_socket):
    config = {"recon_server_address": RECON_SERVER_ADDRESS}
    # Call the function with missing port in config
    # It should not raise an exception or try to connect
    send_to_recon_server(b"test data", config)

    # Ensure that socket methods are not called due to missing port
    mock_socket.assert_not_called()


# Test for send_to_recon_server with missing address and port in config
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_missing_address_and_port(mock_socket):
    config = {}
    # Call the function with missing address and port in config
    # It should not raise an exception or try to connect
    send_to_recon_server(b"test data", config)

    # Ensure that socket methods are not called due to missing address and port
    mock_socket.assert_not_called()


# Test for send_to_recon_server with invalid config types
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_invalid_config_types(mock_socket):
    config = {
        "recon_server_address": None,  # Invalid address
        "recon_server_port": None,  # Invalid port
    }
    # Call the function with invalid config
    # It should not raise an exception or try to connect
    send_to_recon_server(b"test data", config)

    # Ensure that socket methods are not called due to invalid config types
    mock_socket.assert_not_called()


# Test for send_to_recon_server with empty buffer
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_empty_buffer(mock_socket, sample_config):
    # Create a mock socket instance
    mock_socket_instance = MagicMock()
    mock_socket.return_value.__enter__.return_value = mock_socket_instance

    # Empty buffer data
    optional_buffer = b""

    # Call the function
    send_to_recon_server(optional_buffer, sample_config)

    # Check that the socket was created with the correct arguments
    mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)

    # Check that the connect method was called with the correct arguments
    mock_socket_instance.connect.assert_called_once_with(
        (RECON_SERVER_ADDRESS, RECON_SERVER_PORT)
    )

    # Check that the sendall method was called with the empty buffer
    mock_socket_instance.sendall.assert_called_once_with(optional_buffer)


# Test for send_to_recon_server with a valid configuration but no buffer
@patch("pulseforge.server._server.socket.socket")
def test_send_to_recon_server_with_no_buffer(mock_socket, sample_config):
    # Create a mock socket instance
    mock_socket_instance = MagicMock()
    mock_socket.return_value.__enter__.return_value = mock_socket_instance

    # No buffer data
    optional_buffer = b""

    # Call the function
    send_to_recon_server(optional_buffer, sample_config)

    # Check that the socket was created with the correct arguments
    mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)

    # Check that the connect method was called with the correct arguments
    mock_socket_instance.connect.assert_called_once_with(
        (RECON_SERVER_ADDRESS, RECON_SERVER_PORT)
    )

    # Check that the sendall method was called with the empty buffer
    mock_socket_instance.sendall.assert_called_once_with(optional_buffer)


@pytest.fixture
def plugins():
    # Mock plugins with example functions
    mock_function = MagicMock(return_value=(b"result_buffer", b"optional_buffer"))
    return {"mock_function": mock_function}


@pytest.fixture
def client_socket():
    return MagicMock()


@pytest.fixture
def main_logger():
    return MagicMock()


@patch("pulseforge.server._server.setup_function_logger")
@patch("pulseforge.server._server.send_to_recon_server")
def test_handle_client_connection_valid_function(
    mock_send_to_recon_server,
    mock_setup_function_logger,
    sample_config,
    plugins,
    client_socket,
    main_logger,
):
    logger = main_logger

    # Mock request data
    request = "mock_function 2 arg1 arg2"
    client_socket.recv.return_value = request.encode("utf-8")

    # Mock function logger
    function_logger = MagicMock()
    mock_setup_function_logger.return_value = function_logger

    # Call the function
    handle_client_connection(sample_config, client_socket, plugins, logger)

    # Check that the function was called correctly
    plugins["mock_function"].assert_called_once_with("arg1", "arg2")

    # Check that logging was done correctly
    logger.info.assert_called_once_with(
        "Calling mock_function with args ['arg1', 'arg2']"
    )
    function_logger.info.assert_called_once_with("Output buffer: b'result_buffer'")

    # Check that the result buffer was sent to the client
    client_socket.sendall.assert_called_once_with(b"result_buffer")

    # Check that the optional buffer was sent to the recon server
    mock_send_to_recon_server.assert_called_once_with(b"optional_buffer", sample_config)


@patch("pulseforge.server._server.setup_function_logger")
@patch("pulseforge.server._server.send_to_recon_server")
def test_handle_client_connection_function_not_found(
    mock_send_to_recon_server,
    mock_setup_function_logger,
    sample_config,
    plugins,
    client_socket,
    main_logger,
):
    logger = main_logger

    # Mock request data for a non-existent function
    request = "non_existent_function 2 arg1 arg2"
    client_socket.recv.return_value = request.encode("utf-8")

    # Call the function
    handle_client_connection(sample_config, client_socket, plugins, logger)

    # Check that the error was logged
    logger.error.assert_called_once_with("Function non_existent_function not found")

    # Ensure no other methods were called
    client_socket.sendall.assert_not_called()
    mock_send_to_recon_server.assert_not_called()
    mock_setup_function_logger.assert_not_called()


@patch("pulseforge.server._server.setup_function_logger")
@patch("pulseforge.server._server.send_to_recon_server")
def test_handle_client_connection_no_optional_buffer(
    mock_send_to_recon_server,
    mock_setup_function_logger,
    sample_config,
    plugins,
    client_socket,
    main_logger,
):
    logger = main_logger

    # Mock plugins with an example function that returns no optional buffer
    def mock_plugin(*args):
        return b"result_buffer", None

    plugins["mock_function"] = MagicMock(side_effect=mock_plugin)

    # Mock request data
    request = "mock_function 2 arg1 arg2"
    client_socket.recv.return_value = request.encode("utf-8")

    # Mock function logger
    function_logger = MagicMock()
    mock_setup_function_logger.return_value = function_logger

    # Call the function
    handle_client_connection(sample_config, client_socket, plugins, logger)

    # Check that the function was called correctly
    plugins["mock_function"].assert_called_once_with("arg1", "arg2")

    # Check that logging was done correctly
    logger.info.assert_called_once_with(
        "Calling mock_function with args ['arg1', 'arg2']"
    )
    function_logger.info.assert_called_once_with("Output buffer: b'result_buffer'")

    # Check that the result buffer was sent to the client
    client_socket.sendall.assert_called_once_with(b"result_buffer")

    # Ensure the optional buffer was not sent to the recon server
    mock_send_to_recon_server.assert_not_called()


@patch("pulseforge.server._server._get_config")
@patch("pulseforge.server._server.setup_main_logger")
@patch("pulseforge.server._server.load_plugins")
@patch("pulseforge.server._server.handle_client_connection")
@patch("pulseforge.server._server.socket.socket")
def test_start_server(
    mock_socket,
    mock_handle_client_connection,
    mock_load_plugins,
    mock_setup_main_logger,
    mock_get_config,
    sample_config,
):
    # Mock the configuration
    mock_get_config.return_value = sample_config

    # Mock the main logger
    mock_logger = MagicMock()
    mock_setup_main_logger.return_value = mock_logger

    # Mock plugins
    mock_plugins = MagicMock()
    mock_load_plugins.return_value = mock_plugins

    # Mock the socket instance and its methods
    mock_socket_instance = MagicMock()
    mock_socket.return_value.__enter__.return_value = mock_socket_instance
    mock_conn = MagicMock()
    mock_addr = ("127.0.0.1", 54321)
    mock_socket_instance.accept.return_value = (mock_conn, mock_addr)

    # Mock to break the infinite loop after the first iteration
    def side_effect(*args, **kwargs):
        raise KeyboardInterrupt

    mock_handle_client_connection.side_effect = side_effect

    # Call the start_server function
    with pytest.raises(KeyboardInterrupt):
        start_server()

    # Check that configuration was loaded
    mock_get_config.assert_called_once()

    # Check that the main logger was set up
    mock_setup_main_logger.assert_called_once()

    # Check that plugins were loaded
    mock_load_plugins.assert_called_once()

    # Check that the socket was created and set up correctly
    mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
    mock_socket_instance.bind.assert_called_once_with(
        (MR_SCANNER_ADDRESS, MR_SCANNER_PORT)
    )
    mock_socket_instance.listen.assert_called_once()

    # Check that the server started listening and accepted a connection
    mock_logger.info.assert_any_call("Server listening on localhost:12345")
    mock_logger.info.assert_any_call(f"Connected by {mock_addr}")

    # Check that the client connection was handled
    mock_handle_client_connection.assert_called_once_with(
        sample_config, mock_conn, mock_plugins, mock_logger
    )
