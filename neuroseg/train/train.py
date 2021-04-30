import logging
from pathlib import Path
from typing import Union

from neuroseg.config import (
    TrainConfig,
    CallbackConfigurator,
    ModelConfigurator,
    OptimizerConfigurator,
    MetricsConfigurator)

from neuroseg.datagens import Datagen
from neuroseg.utils import BatchInspector
from neuroseg.tiledpredict import DataPredictor
from neuroseg.performance_eval import PerformanceEvaluator
from neuroseg.descriptor import RunDescriptorLight
from neuroseg.config import TrainConfig
from neuroseg.datagens import DataGen2D, DataGen3D


def setup_logger(logfile_path: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return


def debug_train_val_datagens(config: TrainConfig,
                             train_datagen: Union[DataGen2D, DataGen3D],
                             val_datagen: Union[DataGen2D, DataGen3D]):
    """datagens debug batch inspector"""
    if config.train_datagen_inspector:
        train_batch = next(train_datagen.data.__iter__())
        _ = BatchInspector(config, train_batch)
    if config.val_datagen_inspector:
        val_batch = next(val_datagen.data.__iter__())
        _ = BatchInspector(config, val_batch)


def train(train_config: TrainConfig):
    setup_logger(train_config.logfile_path)
    train_datagen = Datagen(train_config,
                            partition="train",
                            normalize_inputs=True,
                            verbose=False,
                            data_augmentation=True)

    val_datagen = Datagen(train_config,
                          partition="val",
                          normalize_inputs=True,
                          verbose=False,
                          data_augmentation=False)

    debug_train_val_datagens(train_config, train_datagen, val_datagen)

    callback_cfg = CallbackConfigurator(train_config)
    callbacks = callback_cfg.callbacks

    model_cfg = ModelConfigurator(train_config)
    model = model_cfg.model

    optimizer_cfg = OptimizerConfigurator(train_config)

    metrics_cfg = MetricsConfigurator(train_config)

    model.compile(optimizer=optimizer_cfg.optimizer,
                  loss=metrics_cfg.loss,
                  metrics=metrics_cfg.track_metrics)

    model_history = model.fit(
        x=train_datagen.data,
        steps_per_epoch=train_datagen.steps_per_epoch,
        validation_data=val_datagen.data,
        validation_steps=val_datagen.steps_per_epoch,
        epochs=train_config.epochs,
        callbacks=callbacks)

    model.save(str(train_config.final_model_path))

    if train_config.evaluate_performance:
        dp = DataPredictor(train_config, model)
        ev = PerformanceEvaluator(train_config, dp.predicted_data)
        performance_dict = ev.measure_dict
    else:
        performance_dict = None

    _ = RunDescriptorLight(train_config,
                           performance_metrics_dict=performance_dict,
                           model_history_dict=model_history)

    return model, model_history, performance_dict
