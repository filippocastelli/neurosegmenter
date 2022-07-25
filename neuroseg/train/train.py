import logging
from pathlib import Path
from typing import Union
from contextlib import nullcontext

import yaml

from neuroseg.config import (
    TrainConfig,
    CallbackConfigurator,
    ModelConfigurator,
    OptimizerConfigurator,
    MetricsConfigurator,
    WandbConfigurator)

import tensorflow as tf

from neuroseg.datagens import Datagen
from neuroseg.utils import BatchInspector
from neuroseg.tiledpredict import DataPredictor
from neuroseg.performance_eval import PerformanceEvaluator
from neuroseg.performance_eval import InstanceSegmentationPerformanceEvaluator
from neuroseg.descriptor import RunDescriptorLight
from neuroseg.config import TrainConfig
from neuroseg.datagens import DataGen2D, DataGen3D
from neuroseg.instance_segmentation import VoronoiInstanceSegmenter



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
        _ = BatchInspector(config, train_batch, title="TRAIN DATAGEN BATCH DEBUG")
    if config.val_datagen_inspector:
        val_batch = next(val_datagen.data.__iter__())
        _ = BatchInspector(config, val_batch, title="VAL DATAGEN BATCH DEBUG")


def get_strategy_scope(config: TrainConfig):
    strategy_name = config.distribute_strategy
    if strategy_name == "mirrored":
        return tf.distribute.MirroredStrategy().scope()
    else:
        return nullcontext()


def dump_custom_objects(config: TrainConfig,
                        optimizer_cfg: OptimizerConfigurator,
                        metrics_cfg: MetricsConfigurator):
    custom_object_names = {
        "optimizer": optimizer_cfg.optimizer_name,
        "loss": metrics_cfg.loss_name,
        "metrics": metrics_cfg.track_metrics_names
    }

    with config.custom_objects_path.open(mode="w") as custom_objects_dump:
        yaml.dump(custom_object_names, custom_objects_dump)


def train(train_config: TrainConfig):
    wc = WandbConfigurator(train_config)

    tf.random.set_seed(train_config.da_seed)
    tf.keras.utils.set_random_seed(train_config.da_seed)
    tf.config.experimental.enable_op_determinism()

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

    with get_strategy_scope(train_config):
        model_cfg = ModelConfigurator(train_config)
        model = model_cfg.model
        optimizer_cfg = OptimizerConfigurator(train_config)
        metrics_cfg = MetricsConfigurator(train_config)

    dump_custom_objects(train_config, optimizer_cfg, metrics_cfg)
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
        inseg = VoronoiInstanceSegmenter(config=train_config, predicted_data=dp.predicted_data)
        ev = PerformanceEvaluator(train_config, dp.predicted_data)
        performance_dict = ev.measure_dict
        instance_performance_ev = InstanceSegmentationPerformanceEvaluator(
            ground_truth=None,
            instance_segmentation=inseg.segmented_dict,
            config=train_config
        )
        instance_performance_dict = instance_performance_ev.metrics_dict
        instance_performance_aggregated_metrics = instance_performance_ev.aggregated_metrics_dict

        voxel_performance_dict = {"voxel_performance_"+ key: val for key, val in performance_dict.items()}
        voxel_performance_aggregated_dict = {"voxel_performance_aggregated_"+ key: val for key, val in ev.metrics.items()}
        instance_performance_dict = {"instance_performance_" + key: val for key, val in instance_performance_dict.items()}
        instance_performance_aggregated_dict = {"instance_performance_aggregated_"+ key: val for key, val in instance_performance_aggregated_metrics.items()}
        
        wc.log_metrics(voxel_performance_dict)
        wc.log_metrics(voxel_performance_aggregated_dict)
        wc.log_metrics(instance_performance_dict)
        wc.log_metrics(instance_performance_aggregated_dict)
    else:
        performance_dict = None

    _ = RunDescriptorLight(train_config,
                           performance_metrics_dict=performance_dict,
                           model_history_dict=model_history)

    return model, model_history, performance_dict
