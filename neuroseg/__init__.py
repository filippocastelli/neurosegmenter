from neuroseg.training import train
from neuroseg.predict import predict
from neuroseg.config  import TrainConfig
from neuroseg.config import PredictConfig

with open("neuroseg/version", "r") as fh:
	__version__ = fh.read()
#__version__ = "0.0.1"
