from argparse import ArgumentParser
from pathlib import Path
import logging

from neuroseg import PredictConfig, predict


def setup_logger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)


def main():
    parser = ArgumentParser()

    parser.add_argument("-c", "--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/tests/test_train_cfg_3d.yml",
                        help="Configuration file path")

    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.configuration_path_str)

    config = PredictConfig(cfg_path)
    _, _, _ = predict(config)


if __name__ == "__main__":
    main()
