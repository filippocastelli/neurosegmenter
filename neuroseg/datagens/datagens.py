from datagens.datagen2d import dataGen2D

def get_datagen(config, partition="train",
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
    else:
        raise NotImplementedError(config.training_mode)