from typing import Union
from neuroseg.metrics import jaccard_index, dice_coefficient, weighted_cross_entropy_loss, weighted_categorical_crossentropy_loss
from neuroseg.config import TrainConfig, PredictConfig
import numpy as np

class MetricsConfigurator:

    def __init__(self,
                 config: Union[TrainConfig, PredictConfig]) -> None:

        self.config = config
        self.track_metrics_names = self.config.track_metrics
        self.loss_name = self.config.loss

        self.pos_weight = self.config.pos_weight if self.config.pos_weight is not None else 1.0
        self.class_weights = self.config.class_weights if self.config.class_weights is not None else np.array([1., 1., 1.])

        self.class_weights = np.array(self.class_weights).astype(np.float64)

        self.track_metrics = self._get_track_metrics(self.track_metrics_names)
        self.loss = self._get_metric(self.loss_name)

    def _get_metric(self, metric_name: str):
        SUPPORTED_METRICS = {
            "jaccard_index": jaccard_index,
            "dice_coefficient": dice_coefficient,
            "binary_crossentropy": "binary_crossentropy",
            "accuracy": "accuracy",
            "weighted_binary_crossentropy": weighted_cross_entropy_loss(self.pos_weight),
            "weighted_categorical_crossentropy": weighted_categorical_crossentropy_loss(self.class_weights)
        }

        if metric_name in SUPPORTED_METRICS:
            return SUPPORTED_METRICS[metric_name]
        else:
            raise NotImplementedError(metric_name)

    def _get_track_metrics(self, track_metric_list: Union[list, tuple]) -> list:
        metric_list = []
        for track_metric_name in track_metric_list:
            metric = self._get_metric(track_metric_name)
            metric_list.append(metric)
        return metric_list
