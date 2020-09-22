import yaml

def load_yml(conf_path):
    """load yml config files"""
    with conf_path.open(mode="r") as in_conf_file:
        cfg = yaml.load(in_conf_file, Loader=yaml.FullLoader)
    return cfg

def save_yml(conf_dict, output_path):
    """save yml config files"""
    with output_path.open(mode="w") as out_conf_file:
        yaml.dump(conf_dict, out_conf_file)
    return