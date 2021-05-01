from typing import Union

from tensorflow.keras.optimizers import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Ftrl,
    Nadam,
    RMSprop,
    SGD
)

from neuroseg.config import TrainConfig, PredictConfig


class OptimizerConfigurator:
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig]):
        self.config = config
        self.optimizer_cfg = self.config.optimizer_cfg
        self.optimizer_name = self.optimizer_cfg["optimizer"]
        self.optimizer = self._configure_optimizer()

    @staticmethod
    def _get_optimizer_arg_dict(optimizer_cfg_dict):
        cfg_copy = optimizer_cfg_dict.copy()
        cfg_copy.pop("optimizer")
        return cfg_copy

    @staticmethod
    def _get_optimizer(optimizer_name: str):

        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "ftrl": Ftrl,
            "nadam": Nadam,
            "rmsprop": RMSprop,
            "sgd": SGD
        }

        if optimizer_name in SUPPORTED_OPTIMIZERS:
            return SUPPORTED_OPTIMIZERS[optimizer_name]
        else:
            raise NotImplementedError(optimizer_name)

    def _configure_optimizer(self):
        optimizer_cls = self._get_optimizer(self.optimizer_name)
        optimizer_args = self._get_optimizer_arg_dict(self.optimizer_cfg)

        return optimizer_cls(**optimizer_args)
