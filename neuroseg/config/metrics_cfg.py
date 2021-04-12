from neuroseg.metrics import jaccard_index, dice_coefficient

class MetricsConfigurator:
    
    def __init__(self,
                 config):
        
        self.config = config
        self.track_metrics_names = self.config.track_metrics
        self.loss_name = self.config.loss
        
        self.track_metrics = self._get_track_metrics(self.track_metrics_names)
        self.loss = self._get_metric(self.loss_name)
    
    @staticmethod
    def _get_metric(metric_name):
        SUPPORTED_METRICS = {
            "jaccard_index": jaccard_index,
            "dice_coefficient": dice_coefficient,
            "binary_crossentropy": "binary_crossentropy",
            "accuracy": "accuracy",
            }
        
        if metric_name in SUPPORTED_METRICS: 
            return SUPPORTED_METRICS[metric_name]
        else:
            raise NotImplementedError(metric_name)
            
    @classmethod
    def _get_track_metrics(cls, track_metric_list):
        metric_list = []
        for track_metric_name in track_metric_list:
            metric = cls._get_metric(track_metric_name)
            metric_list.append(metric)
            