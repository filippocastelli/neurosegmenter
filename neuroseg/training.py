from argparse import ArgumentParser
from pathlib import Path
import logging

from config import (
    TrainConfig,
    CallbackConfigurator,
    ModelConfigurator,
    OptimizerConfigurator,
    MetricsConfigurator)

from datagens import Datagen
from utils import BatchInspector
from tiledpredict import DataPredictor
from performance_eval import PerformanceEvaluator
from descriptor import RunDescriptorLight

def setup_logger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    
def debug_train_val_datagens(config, train_datagen, val_datagen):
    
    if config.train_datagen_inspector:
        train_batch = next(train_datagen.data.__iter__())
        _ = BatchInspector(config, train_batch)
    if config.val_datagen_inspector:
        val_batch = next(val_datagen.data.__iter__())
        _ = BatchInspector(config, val_batch)
        
def main(cfg_path):
    
    config = TrainConfig(cfg_path)
    setup_logger(config.logfile_path)
    train_datagen = Datagen(config,
                              partition="train",
                              normalize_inputs=True,
                              ignore_last_channel=True,
                              verbose=False,
                              data_augmentation=True)
    
    val_datagen = Datagen(config,
                              partition="val",
                              normalize_inputs=True,
                              ignore_last_channel=True,
                              verbose=False,
                              data_augmentation=False)
    
    debug_train_val_datagens(config, train_datagen, val_datagen)
    
    callback_cfg = CallbackConfigurator(config)
    callbacks = callback_cfg.callbacks
    
    model_cfg = ModelConfigurator(config)
    model = model_cfg.model
    
    optimizer_cfg = OptimizerConfigurator(config)

    metrics_cfg = MetricsConfigurator(config)
    
    model.compile(optimizer=optimizer_cfg.optimizer,
                  loss=metrics_cfg.loss,
                  metrics=metrics_cfg.track_metrics)
    
    model_history = model.fit(
        x=train_datagen.data,
        steps_per_epoch = train_datagen.steps_per_epoch,
        validation_data=val_datagen.data,
        validation_steps=val_datagen.steps_per_epoch,
        epochs=config.epochs,
        callbacks=callbacks)
    
    model.save(str(config.final_model_path))
    
    if config.evaluate_performance:
        dp = DataPredictor(config, model)
        ev = PerformanceEvaluator(config, dp.predicted_data)
        performance_dict = ev.measure_dict
    else:
        performance_dict = None
        
    _ = RunDescriptorLight(config,
                               performance_metrics_dict=performance_dict,
                               model_history_dict=model_history)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/tests/test_train_cfg_3d.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    cfg_path = Path(args.configuration_path_str)
    
    main(cfg_path)
    
    

        
