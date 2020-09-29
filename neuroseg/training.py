from argparse import ArgumentParser
from pathlib import Path
import logging

from config import TrainConfig
from datagens import get_datagen

def setup_logger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
def main(cfg_path):
    
    config = TrainConfig(cfg_path)
    setup_logger(config.logfile_path)
    datagen_obj = get_datagen(config, partition="train", normalize=True, verbose=False)
    data = datagen_obj.data
    data_iterator = data.as_numpy_iterator()
    
    ex_list = list(data_iterator)
    
    print("debug")

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/config/ex_cfg.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    cfg_path = Path(args.configuration_path_str)
    
    main(cfg_path)
    

    