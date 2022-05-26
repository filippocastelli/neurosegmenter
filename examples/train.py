from pathlib import Path
from wandb import Config
import yaml
from argparse import ArgumentParser

from neuroseg import TrainConfig, train

def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, dest="config_path", help="config_path")
    
    args = parser.parse_args()
    config_path = Path(args.config_path)

    config = TrainConfig(yml_path=config_path)
    train(config)

if __name__ == "__main__":
    main()