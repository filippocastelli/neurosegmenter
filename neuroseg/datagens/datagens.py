from neuroseg.datagens.datagen2d import DataGen2D
from neuroseg.datagens.datagen3d import DataGen3D

def Datagen(config, partition="train",
                normalize_inputs=True,
                verbose=False,
                data_augmentation=False):
    
    if config.training_mode == "2d":
        return DataGen2D(config=config,
                         partition=partition,
                         normalize_inputs=normalize_inputs,
                         verbose=verbose,
                         data_augmentation=data_augmentation)
    elif config.training_mode == "3d":
        return DataGen3D(config=config,
                         partition=partition,
                         data_augmentation=data_augmentation,
                         verbose=verbose,
                         normalize_inputs=normalize_inputs)
    else:
        raise NotImplementedError(config.training_mode)