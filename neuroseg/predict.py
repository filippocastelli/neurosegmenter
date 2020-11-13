from argparse import ArgumentParser
from pathlib import Path
import logging 


from config import PredictConfig
from utils import load_volume, save_volume
from tiledpredict import DataPredictor2D


def main(cfg_path):
    
    config = PredictConfig(cfg_path)
    # setup_logger(config.logfile_path)
    dp = DataPredictor2D(config)
    
    
    
    print("ciao")

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/tests/test_predict_cfg.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    cfg_path = Path(args.configuration_path_str)
    
    main(cfg_path)