from pathlib import Path
import yaml
from pathlib import Path
import yaml
import tifffile

from neuroseg import PredictConfig, predict


print(Path.cwd())
config_path = Path("config_predict.yml")

with config_path.open(mode="r") as infile:
    yaml.load(infile, Loader=yaml.FullLoader)
config = PredictConfig(config_path)

predicted_data = predict(config, in_fpath=Path("/home/phil/substack"))