"""
"""
import socket
import os
import importlib.util
import logging
from datetime import datetime
import pickle
import requests
import json

# Load configuration
CONFIG_FILE_PATH = os.getenv('CONFIG_FILE_PATH', './server_config.json')
with open(CONFIG_FILE_PATH, 'r') as config_file:
    config = json.load(config_file)

# Extract server configuration
HOST = config['server']['host']
PORT = config['server']['port']

# Extract secondary server URL
SECONDARY_SERVER_URL = config['secondary_server']['url']

# Extract plugin and log directories
PLUGIN_DIR = config.get('plugin_dir', './plugins')
LOG_DIR = config.get('log_dir', '.')

# Extract client configuration
CLIENT_HOST = config['client']['host']
CLIENT_PORT = config['client']['port']

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure main session logging
session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_filename = os.path.join(LOG_DIR, f'session_{session_start_time}.txt')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(main_log_filename), logging.StreamHandler()])
logger = logging.getLogger('main')

def load_plugins(directory):
    plugins = {}
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            filepath = os.path.join(directory, filename)
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugins[module_name] = module
            logger.debug(f"Loaded plugin: {module_name} from {filepath}")
    return plugins

def parse_request(request):
    try:
        # Example format: "funcname n var1 var2 ... varn"
        parts = request.split()
        function_name = parts[0]
        n = int(parts[1])
        args = parts[2:2 + n]
        logger.debug(f"Parsed request - Function: {function_name}, Args: {args}")
        return function_name, args
    except Exception as e:
        logger.error(f"Failed to parse request: {e}")
        return None, None

def setup_function_logger(function_name):
    function_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    function_log_filename = os.path.join(LOG_DIR, f'{function_name}_{function_start_time}.txt')
    function_logger = logging.getLogger(function_name)
    function_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(function_log_filename)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    function_logger.addHandler(handler)
    return function_logger

def send_file_to_secondary_server(file_path):
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(SECONDARY_SERVER_URL, files=files)
            response.raise_for_status()
            logger.info(f"Successfully sent file {file_path} to secondary server.")
    except Exception as e:
        logger.error(f"Failed to send file {file_path} to secondary server: {e}")

def handle_client(connection, plugins):
    try:
        # Receive data from the client
        request = connection.recv(1024).decode('utf-8')
        logger.info(f"Received request: {request}")
        function_name, args = parse_request(request)
        if function_name in plugins:
            func = getattr(plugins[function_name], function_name, None)
            if func:
                function_logger = setup_function_logger(function_name)
                logger.info(f"Calling {function_name}.py with args {args}")
                function_logger.info(f"Function {function_name} called with args: {args}")

                # Convert string arguments to appropriate types if necessary
                args = [int(arg) for arg in args]
                result, file_path = func(*args)
                function_logger.info(f"Function {function_name} returned: {result}")
                function_logger.info(f"Function {function_name} generated file: {file_path}")

                # Serialize the result to a buffer
                result_buffer = pickle.dumps(result)
                connection.sendall(result_buffer)

                # Send the file to the secondary server
                if file_path and os.path.exists(file_path):
                    send_file_to_secondary_server(file_path)
            else:
                logger.warning(f"Function {function_name} not found in plugin")
                connection.sendall(b'Function not found')
        else:
            logger.warning(f"Function {function_name} not available")
            connection.sendall(b'Function not available')
    except Exception as e:
        logger.error(f"Error handling client: {e}")
        connection.sendall(f'Error: {e}'.encode('utf-8'))
    finally:
        connection.close()
        logger.info("Closed connection with client")

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f'Server listening on {HOST}:{PORT}')
        plugins = load_plugins(PLUGIN_DIR)
        while True:
            conn, addr = s.accept()
            with conn:
                logger.info(f'Connected by {addr}')
                handle_client(conn, plugins)

if __name__ == "__main__":
    start_server()
