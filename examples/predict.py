from pathlib import Path
import yaml

from neuroseg import PredictConfig, predict

print(Path.cwd())
config_path = Path("predict_config.yml")

with config_path.open(mode="r") as infile:
    yaml.load(infile, Loader=yaml.FullLoader)
config = PredictConfig(config_path)

predict(config)