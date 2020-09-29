from datagens.datagen2d import dataGen2D

def get_datagen(config, partition="train", normalize=True, verbose=False):
    if config.training_mode == "2d":
        return dataGen2D(config, partition, normalize,verbose)
    else:
        raise NotImplementedError(config.training_mode)