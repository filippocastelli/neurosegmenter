from argparse import ArgumentParser
from pathlib import Path

import yaml
from tensorflow.keras.models import load_model

from neuroseg.config import PredictConfig
from neuroseg.tiledpredict import DataPredictor
from neuroseg.instance_segmentation import InstanceSegmenter

def predict(predict_config: PredictConfig, in_fpath=None):
    model_fpath = predict_config.model_path
    model = load_model(str(model_fpath), compile=False)
    dp = DataPredictor(predict_config, model=model)
    inseg = InstanceSegmenter(predict_config, dp.predicted_data)
    # ev = PerformanceEvaluator(predict_config, dp.predicted_data)
    # performance_dict = ev.measure_dict
    # _ = RunDescriptorLight(predict_config, performance_metrics_dict=performance_dict)
    if inseg.enable_instance_segmentation:
        return dp.predicted_data, inseg.segmented_vol, inseg.segmented_vol_rgb
    else:
        return dp.predicted_data
