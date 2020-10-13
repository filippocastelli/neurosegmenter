from models import ResUNET2D
from config.optimizer_cfg import OptimizerConfigurator

class ModelConfigurator:
    def __init__(self,
                 config,
                 compile_model=True):
        self.config = config
        self.model_name = self.config.model
        self.loss = self.config.loss
        self.track_metrics = self.config.track_metrics
        self.optimizer_cfg = OptimizerConfigurator(self.config)
        self.optimizer = self.optimizer_cfg.optimizer
        
        self.model_cfg = self._get_model_cfg()
        self.model = self.model_cfg.model
        
        if self.compile_model:
            self.compile_model()
    
    def _get_model_cfg(self): #maybe change to dict?
        if self.model_name=="resunet2d":
            return ResUNET2D(self.config)
        else:
            raise NotImplementedError(self.model_name)
            
    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss, #TODO LossConfigurator
            metrics=self.track_metrics)