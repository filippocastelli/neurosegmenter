from pathlib import Path
from yml_io import load_yml, save_yml

def load_cfg(cfg_path):
    cfg_dict = load_yml(cfg_path)
    for key, pathstring in cfg_dict["paths"].items():
        cfg_dict["paths"][key] = decode_path(pathstring)
    
    return cfg_dict

def save_cfg(cfg_dict, cfg_path):
    for key, pathlib_path in cfg_dict["paths"].items():
        cfg_dict["paths"][key] = encode_path(pathlib_path)
    
    save_yml(cfg_dict, cfg_path)

def decode_path(path_string):
    if path_string == "":
        parsed_path = None
    else:
        parsed_path = Path(path_string)
    return parsed_path

def encode_path(pathlib_path):
    if pathlib_path is None:
        encoded_path = None
    else:
        encoded_path = str(pathlib_path)
    return encoded_path