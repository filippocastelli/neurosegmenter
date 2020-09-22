from argparse import ArgumentParser
from pathlib import Path

from config import load_yml, load_paths


def main(cfg):
    
    paths = load_paths(cfg)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="config.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    conf_path = Path(args.configuration_path_str)
    
    yml_cfg = load_yml(conf_path)
    
    main(yml_cfg)