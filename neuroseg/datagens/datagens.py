from datagens.datagen2d import dataGen2D
from datagens.datagen3d import datagen3DSingle

def Datagen(config, partition="train",
                normalize_inputs=True,
                ignore_last_channel=False,
                verbose=False,
                data_augmentation=False):
    
    if config.training_mode == "2d":
        return dataGen2D(config=config,
                         partition=partition,
                         normalize_inputs=normalize_inputs,
                         ignore_last_channel=ignore_last_channel,
                         verbose=verbose,
                         data_augmentation=data_augmentation)
    elif config.training_mode == "3d":
        
        return datagen3DSingle(config=config,
                               partition=partition,
                               data_augmentation=data_augmentation,
                               verbose=verbose,
                               normalize_inputs=normalize_inputs)
    
        raise NotImplementedError("3d")
        # single images
        # single stack
        # multi stack
    else:
        raise NotImplementedError(config.training_mode)