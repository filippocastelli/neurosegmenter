import inspect
from pathlib import Path

from neuroseg.training import train
from neuroseg.predict import predict
from neuroseg.config  import TrainConfig
from neuroseg.config import PredictConfig

import neuroseg
version_path = Path(inspect.getfile(neuroseg)).parent.joinpath("version")

with version_path.open(mode="r") as fh:
	__version__ = fh.read()
    
#__version__ = "0.0.1"
