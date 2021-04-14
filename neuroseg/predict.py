from argparse import ArgumentParser
from pathlib import Path


from neuroseg.config import PredictConfig
from neuroseg.tiledpredict import DataPredictor
from neuroseg.performance_eval import PerformanceEvaluator

from neuroseg.descriptor import RunDescriptorLight


def predict(predict_config: PredictConfig):
    dp = DataPredictor(predict_config)
    ev = PerformanceEvaluator(predict_config, dp.predicted_data)
    performance_dict = ev.measure_dict
    _ = RunDescriptorLight(predict_config, performance_metrics_dict=performance_dict)
    return performance_dict