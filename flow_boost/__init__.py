from pathlib import Path
import flow_boost
import os
import configparser

# constants
#reffolder = Path(flow_boost.__file__).parent.parent
# print("file: ", flow_boost.__file__)
reffolder = Path(__file__).parent.parent
CONFIG_FILE  = os.path.join(reffolder, "config.cfg")

def load_config(file_path):
    """
    Loads the configuration file.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    else:
        print(f"[Load] Loaded configuration file {file_path}!")
    config.read(file_path, encoding="utf8")
    return config

# Load the configuration
try:
    # print(f"Module imported in PID: {os.getpid()}")
    # print(f"name is {__name__}")
    import os
    cfg = load_config(CONFIG_FILE)
    if hasattr(cfg, "_runtime_variables"):
        print("length: ", len(cfg._runtime_variables))
except FileNotFoundError as e:
    print(e)
    exit()