from typing import Union

from neuroseg.datagens.datagen2d import DataGen2D
from neuroseg.datagens.datagen3d import DataGen3D
from neuroseg.config import TrainConfig, PredictConfig


def Datagen(config: Union[TrainConfig, PredictConfig],
            partition: str = "train",
            normalize_inputs: bool = True,
            verbose: bool = False,
            data_augmentation: bool = False):
    if config.training_mode == "2d":
        return DataGen2D(config=config,
                         partition=partition,
                         verbose=verbose,
                         data_augmentation=data_augmentation)
    elif config.training_mode == "3d":
        return DataGen3D(config=config,
                         partition=partition,
                         data_augmentation=data_augmentation,
                         verbose=verbose)
    else:
        raise NotImplementedError(config.training_mode)
