from pathlib import Path
import yaml

from neuroseg import TrainConfig, train

print(Path.cwd())
#config_path = Path("/opt/examples/epfl_em_3d.yml")
config_path = Path("epfl_em_3d.yml")
with config_path.open(mode="r") as infile:
    yaml.load(infile, Loader=yaml.FullLoader)
config = TrainConfig(config_path)

train(config)
