from pathlib import Path

from neuroseg.tiledpredict.tp2d import DataPredictor2D, MultiVolumeDataPredictor2D, ChunkDataPredictor2D
from neuroseg.tiledpredict.tp3d import DataPredictor3D, MultiVolumeDataPredictor3D, H5DataPredictor
from neuroseg.tiledpredict.tp2d_single_images import SingleImagesDataPredictor

def DataPredictor(config, model=None, in_fpath: Path = None):
    config_type = config.config_type
    mode = config.training_mode if config_type == "training" else config.prediction_mode
    data_mode = config.dataset_mode if config_type == "training" else config.data_mode
    
    if mode == "2d":
        if data_mode == "single_images":
            return SingleImagesDataPredictor(config, model, in_fpath)
        elif data_mode in ["stack"]:
            return DataPredictor2D(config, model, in_fpath)
        elif data_mode == "multi_stack":
            return MultiVolumeDataPredictor2D(config, model)
        elif data_mode == "zetastitcher":
            return ChunkDataPredictor2D(config, model)
        else:
            raise NotImplementedError(data_mode)
        
    elif mode == "3d":
        if data_mode in ["single_images", "stack"]:
            return DataPredictor3D(config, model)
        elif data_mode in ["multi_stack"]:
            return MultiVolumeDataPredictor3D(config, model)
        elif data_mode == "h5_dataset":
            return H5DataPredictor(config, model)
        else:
            raise NotImplementedError(data_mode)
            
    else:
        raise NotImplementedError(mode)
    
    