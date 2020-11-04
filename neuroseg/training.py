from argparse import ArgumentParser
from pathlib import Path
import logging

from config import (
    TrainConfig,
    CallbackConfigurator,
    ModelConfigurator,
    OptimizerConfigurator,
    MetricsConfigurator)

from datagens import get_datagen
from utils import BatchInspector2D

def setup_logger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
def main(cfg_path):
    
    config = TrainConfig(cfg_path)
    setup_logger(config.logfile_path)
    train_datagen = get_datagen(config,
                              partition="train",
                              normalize_inputs=True,
                              ignore_last_channel=True,
                              verbose=False,
                              data_augmentation=True)
    
    val_datagen = get_datagen(config,
                              partition="val",
                              normalize_inputs=True,
                              ignore_last_channel=True,
                              verbose=False,
                              data_augmentation=False)
    
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
        validation_data=val_datagen.data,
        epochs=config.epochs,
        callbacks=callbacks)
    
    model.save(str(config.final_model_path))
    
    
    print("ciao")
    # val_data_debug = val_datagen.data
    # val_iterator = val_data_debug.as_numpy_iterator()
    
    # test_batch = next(val_iterator)
    
    # BatchInspector2D(test_batch)
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/tests/test_train_cfg.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    cfg_path = Path(args.configuration_path_str)
    
    main(cfg_path)
