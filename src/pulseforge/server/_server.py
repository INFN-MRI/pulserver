"""Main sequence design server app."""

__all__ = ["start_server"]

import importlib.util
import yaml
import logging
import os
import pathlib
import socket

from datetime import datetime


# Load configuration
def _get_config():
    CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", None)

    if CONFIG_FILE_PATH is not None:
        with open(CONFIG_FILE_PATH, "r") as config_file:
            config = yaml.safe_load(config_file)
    else:
        config = {}

    return config


# Plugins
def _get_plugin_dir():
    PKG_DIR = pathlib.Path(os.path.realpath(__file__)).parents[1].resolve()
    PLUGIN_DIR = [os.path.join(PKG_DIR, "_apps")]
    CUSTOM_PLUGINS = os.getenv("PULSEFORGE_PLUGINS", None)
    if CUSTOM_PLUGINS:
        PLUGIN_DIR.append(os.path.realpath(CUSTOM_PLUGINS))

    return PLUGIN_DIR


# Logs
def _get_log_dir():
    LOG_DIR = os.getenv("PULSEFORGE_LOG", None)
    if LOG_DIR is None:
        HOME_DIR = pathlib.Path.home()
        LOG_DIR = os.path.join(HOME_DIR, "log")

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    return LOG_DIR


# Configure main session logging
def setup_main_logger():
    LOG_DIR = _get_log_dir()
    session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_filename = os.path.join(LOG_DIR, f"session_{session_start_time}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(main_log_filename), logging.StreamHandler()],
    )
    logger = logging.getLogger("main")
    return logger


def setup_function_logger(function_name):
    LOG_DIR = _get_log_dir()
    function_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    function_log_filename = os.path.join(
        LOG_DIR, f"{function_name}_{function_start_time}.log"
    )
    function_logger = logging.getLogger(function_name)
    function_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(function_log_filename)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    function_logger.addHandler(handler)
    return function_logger


def load_plugins(logger):
    # Get plugin path
    PLUGIN_DIR = _get_plugin_dir()

    # Load plugins
    plugins = {}
    for directory in PLUGIN_DIR:
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                filepath = os.path.join(directory, filename)
                module_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                plugins[module_name] = module
                logger.debug(f"Loaded plugin: {module_name} from {filepath}")
    return plugins


def parse_request(request, logger):
    try:
        # Example format: "funcname n var1 var2 ... varn"
        parts = request.split()
        function_name = parts[0]
        n = int(parts[1])
        args = parts[2 : 2 + n]
        logger.debug(f"Parsed request - Function: {function_name}, Args: {args}")
        return function_name, args
    except Exception as e:
        logger.error(f"Failed to parse request: {e}")
        return None, None


def send_to_recon_server(optional_buffer, config):
    RECON_SERVER_ADDRESS = config.get("recon_server_address", None)
    RECON_SERVER_PORT = config.get("recon_server_port", None)
    if RECON_SERVER_ADDRESS is not None and RECON_SERVER_PORT is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RECON_SERVER_ADDRESS, RECON_SERVER_PORT))
            s.sendall(optional_buffer)


def handle_client_connection(config, client_socket, plugins, logger):
    request = client_socket.recv(1024).decode("utf-8")
    function_name, args = parse_request(request, logger)
    if function_name in plugins:
        # Select function
        function = plugins[function_name]

        # Set-up logging
        function_logger = setup_function_logger(function_name)
        logger.info(f"Calling {function_name} with args {args}")

        # Run design function
        result_buffer, optional_buffer = function(*args)

        # Log the output to the function-specific log file
        function_logger.info(f"Output buffer: {result_buffer}")

        # Send the result buffer to the client
        client_socket.sendall(result_buffer)

        # Optionally send the reconstruction info to the secondary server
        if optional_buffer is not None:
            send_to_recon_server(optional_buffer, config)
    else:
        logger.error(f"Function {function_name} not found")


def start_server():
    # Get configuration
    config = _get_config()
    SCANNER_ADDRESS = config.get("scanner_address", None)
    SCANNER_PORT = config.get("scanner_port", None)

    # Set-up main logger
    logger = setup_main_logger()

    # Load plugins
    plugins = load_plugins(logger)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((SCANNER_ADDRESS, SCANNER_PORT))
        s.listen()
        logger.info(f"Server listening on {SCANNER_ADDRESS}:{SCANNER_PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                logger.info(f"Connected by {addr}")
                handle_client_connection(config, conn, plugins, logger)
