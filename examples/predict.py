from pathlib import Path
from argparse import ArgumentParser
from neuroseg import PredictConfig, predict

def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, dest="config_path", help="config_path")
    
    args = parser.parse_args()
    config_path = Path(args.config_path)

    config = PredictConfig(yml_path=config_path)
    predict(config)

if __name__ == "__main__":
    main()