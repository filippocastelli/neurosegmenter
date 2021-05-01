from typing import Union
from neuroseg.models import ResUNET2D, ResUNET3D
from neuroseg.config.optimizer_cfg import OptimizerConfigurator
from neuroseg.config import TrainConfig, PredictConfig


class ModelConfigurator:
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig]) -> None:
        self.config = config
        self.model_name = self.config.model
        self.loss = self.config.loss
        self.track_metrics = self.config.track_metrics
        self.optimizer_cfg = OptimizerConfigurator(self.config)
        self.optimizer = self.optimizer_cfg.optimizer

        self.model_cfg = self._get_model_cfg()
        self.model = self.model_cfg.model
        return

    def _get_model_cfg(self):
        if self.model_name == "resunet2d":
            return ResUNET2D(self.config)
        elif self.model_name == "resunet3d":
            return ResUNET3D(self.config)
        else:
            raise NotImplementedError(self.model_name)
