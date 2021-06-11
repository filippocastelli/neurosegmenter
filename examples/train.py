from pathlib import Path
import yaml

from neuroseg import TrainConfig, train

print(Path.cwd())
config_path = Path("config_2d_single_images.yml")

with config_path.open(mode="r") as infile:
    yaml.load(infile, Loader=yaml.FullLoader)
config = TrainConfig(config_path)

train(config)