
import pytest
from unittest.mock import MagicMock, patch
import logging
import os
import pickle
import json

from pulseforge._server import _server as server

@pytest.fixture
def mock_plugins():
    # Mock the plugins
    plugins = {
        'foo': MagicMock()
    }
    plugins['foo'].foo = MagicMock(return_value=(7, '/tmp/foo_3_4.h5'))
    return plugins

def test_parse_request():
    request = "foo 2 3 4"
    func, args = server.parse_request(request)
    assert func == "foo"
    assert args == ['3', '4']

@patch('socket.socket')
@patch('server.send_file_to_secondary_server')
def test_handle_client(mock_send_file, mock_socket, mock_plugins, tmpdir, caplog):
    # Prepare the mock socket
    mock_conn = MagicMock()
    mock_socket.return_value.accept.return_value = (mock_conn, ('127.0.0.1', 12345))
    mock_conn.recv.return_value = b"foo 2 3 4"
    mock_conn.sendall = MagicMock()

    # Set up a temporary config file
    config_file_content = {
        "server": {
            "host": "localhost",
            "port": 65432
        },
        "secondary_server": {
            "url": "http://localhost:8000/upload"
        },
        "plugin_dir": str(tmpdir.mkdir("plugins")),
        "log_dir": str(tmpdir.mkdir("logs")),
        "client": {
            "host": "localhost",
            "port": 65431
        }
    }
    config_file_path = tmpdir.join("server_config.json")
    with open(config_file_path, 'w') as f:
        f.write(json.dumps(config_file_content))

    os.environ['CONFIG_FILE_PATH'] = str(config_file_path)

    # Set up the plugin directory and add a mock foo.py
    plugin_dir = config_file_content['plugin_dir']
    os.makedirs(plugin_dir, exist_ok=True)
    with open(os.path.join(plugin_dir, 'foo.py'), 'w') as f:
        f.write("""
import numpy as np
import h5py

def foo(var1, var2):
    result = np.max([var1, var2])
    file_path = f"/tmp/foo_{var1}_{var2}.h5"
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('max', data=result)
    return result, file_path
""")

    # Set log directory environment variable
    log_dir = config_file_content['log_dir']
    os.environ['LOG_DIR'] = log_dir
    main_log_file = log_dir.join(f'session_{server.session_start_time}.txt')
    caplog.set_level(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        with patch('server.load_plugins', return_value=mock_plugins):
            with patch('logging.FileHandler', lambda filename: logging.FileHandler(str(log_dir.join(filename)))):
                server.start_server()  # Starts the server and handles the mock client connection

                # Check that the correct function was called
                mock_plugins['foo'].foo.assert_called_with(3, 4)
                # Check that the response was sent back
                expected_result_buffer = pickle.dumps(7)
                mock_conn.sendall.assert_called_with(expected_result_buffer)
                
                # Check that the correct function was called
                mock_plugins['foo'].foo.assert_called_with(3, 4)
                # Check that the response was sent back
                expected_result_buffer = pickle.dumps(7)
                mock_conn.sendall.assert_called_with(expected_result_buffer)
                
                # Check if the function-specific log file is created
                function_log_file = os.path.join(log_dir, f'foo_{server.session_start_time}.txt')
                assert os.path.exists(function_log_file)
                with open(function_log_file) as f:
                    log_content = f.read()
                    assert "Function foo called with args: ['3', '4']" in log_content
                    assert "Function foo returned: 7" in log_content
                    assert "Function foo generated file: /tmp/foo_3_4.h5" in log_content
                
                # Check if the file was sent to the secondary server
                assert mock_send_file.call_count == 1
                assert mock_send_file.call_args[0][0] == '/tmp/foo_3_4.h5'
                
                # Check the content of the main log file
                assert main_log_file.exists()
                with open(main_log_file) as f:
                    log_content = f.read()
                    assert "Received request: foo 2 3 4" in log_content
                    assert "Calling foo.py with args [3, 4]" in log_content
                    assert "Function foo returned: 7" in log_content
                    assert "Function foo generated file: /tmp/foo_3_4.h5" in log_content
                    assert "Closed connection with client" in log_content

@patch('socket.socket')
@patch('server.send_file_to_secondary_server')
def test_handle_client_function_not_found(mock_send_file, mock_socket, mock_plugins, tmpdir, caplog):
    # Prepare the mock socket
    mock_conn = MagicMock()
    mock_socket.return_value.accept.return_value = (mock_conn, ('127.0.0.1', 12345))
    mock_conn.recv.return_value = b"bar 2 3 4"
    mock_conn.sendall = MagicMock()

    # Set up a temporary config file
    config_file_content = {
        "server": {
            "host": "localhost",
            "port": 65432
        },
        "secondary_server": {
            "url": "http://localhost:8000/upload"
        },
        "plugin_dir": str(tmpdir.mkdir("plugins")),
        "log_dir": str(tmpdir.mkdir("logs")),
        "client": {
            "host": "localhost",
            "port": 65431
        }
    }
    config_file_path = tmpdir.join("server_config.json")
    with open(config_file_path, 'w') as f:
        f.write(json.dumps(config_file_content))

    os.environ['CONFIG_FILE_PATH'] = str(config_file_path)

    # Set log directory environment variable
    log_dir = config_file_content['log_dir']
    os.environ['LOG_DIR'] = log_dir
    main_log_file = os.path.join(log_dir, f'session_{server.session_start_time}.txt')
    caplog.set_level(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        with patch('server.load_plugins', return_value=mock_plugins):
            with patch('logging.FileHandler', lambda filename: logging.FileHandler(str(log_dir.join(filename)))):
                server.start_server()  # Starts the server and handles the mock client connection

                # Check that the function was not called
                mock_plugins['foo'].foo.assert_not_called()
                # Check that the 'Function not available' message was sent back
                mock_conn.sendall.assert_called_with(b'Function not available')
                # Verify logs
                assert "Received request: bar 2 3 4" in caplog.text
                assert "Function bar not available" in caplog.text

    # Check if main log file is created
    assert os.path.exists(main_log_file)
    with open(main_log_file) as f:
        log_content = f.read()
        assert "Received request: bar 2 3 4" in log_content
        assert "Function bar not available" in log_content

@patch('socket.socket')
@patch('server.send_file_to_secondary_server')
def test_handle_client_invalid_request(mock_send_file, mock_socket, mock_plugins, tmpdir, caplog):
    # Prepare the mock socket
    mock_conn = MagicMock()
    mock_socket.return_value.accept.return_value = (mock_conn, ('127.0.0.1', 12345))
    mock_conn.recv.return_value = b"invalid request format"
    mock_conn.sendall = MagicMock()

    # Set up a temporary config file
    config_file_content = {
        "server": {
            "host": "localhost",
            "port": 65432
        },
        "secondary_server": {
            "url": "http://localhost:8000/upload"
        },
        "plugin_dir": str(tmpdir.mkdir("plugins")),
        "log_dir": str(tmpdir.mkdir("logs")),
        "client": {
            "host": "localhost",
            "port": 65431
        }
    }
    config_file_path = tmpdir.join("server_config.json")
    with open(config_file_path, 'w') as f:
        f.write(json.dumps(config_file_content))

    os.environ['CONFIG_FILE_PATH'] = str(config_file_path)

    # Set log directory environment variable
    log_dir = config_file_content['log_dir']
    os.environ['LOG_DIR'] = log_dir
    main_log_file = os.path.join(log_dir, f'session_{server.session_start_time}.txt')
    caplog.set_level(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        with patch('server.load_plugins', return_value=mock_plugins):
            with patch('logging.FileHandler', lambda filename: logging.FileHandler(str(log_dir.join(filename)))):
                server.start_server()  # Starts the server and handles the mock client connection

                # Check that the 'Error' message was sent back
                assert any("Error" in str(call) for call in mock_conn.sendall.call_args_list)
                # Verify logs
                assert "Received request: invalid request format" in caplog.text
                assert "Failed to parse request" in caplog.text

    # Check if main log file is created
    assert os.path.exists(main_log_file)
    with open(main_log_file) as f:
        log_content = f.read()
        assert "Received request: invalid request format" in log_content
        assert "Failed to parse request" in log_content
