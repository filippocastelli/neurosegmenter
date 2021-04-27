from neuroseg.metrics import jaccard_index, dice_coefficient, weighted_cross_entropy_loss


class MetricsConfigurator:

    def __init__(self,
                 config):

        self.config = config
        self.track_metrics_names = self.config.track_metrics
        self.loss_name = self.config.loss

        self.pos_weight = self.config.pos_weight

        self.track_metrics = self._get_track_metrics(self.track_metrics_names)
        self.loss = self._get_metric(self.loss_name)

    def _get_metric(self, metric_name):
        SUPPORTED_METRICS = {
            "jaccard_index": jaccard_index,
            "dice_coefficient": dice_coefficient,
            "binary_crossentropy": "binary_crossentropy",
            "accuracy": "accuracy",
            "weighted_binary_crossentropy": weighted_cross_entropy_loss(self.pos_weight)
        }

        if metric_name in SUPPORTED_METRICS:
            return SUPPORTED_METRICS[metric_name]
        else:
            raise NotImplementedError(metric_name)

    def _get_track_metrics(self, track_metric_list):
        metric_list = []
        for track_metric_name in track_metric_list:
            metric = self._get_metric(track_metric_name)
            metric_list.append(metric)
        return metric_list
