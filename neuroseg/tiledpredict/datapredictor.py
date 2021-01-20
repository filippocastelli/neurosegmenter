from tiledpredict.tp2d import DataPredictor2D
from tiledpredict.tp3d import DataPredictor3D


def DataPredictor(config, model=None):
    config_type = config.config_type
    mode = config.training_mode if config_type == "training" else config.prediction_mode
    
    if mode == "2d":
        return DataPredictor2D(config, model)
    elif mode == "3d":
        return DataPredictor3D(config, model)
    else:
        raise NotImplementedError(mode)
    
    