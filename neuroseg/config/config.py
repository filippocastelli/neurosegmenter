from pathlib import Path
import yaml
from typing import Union
from multiprocessing import cpu_count

from neuroseg.utils import NameGenerator

SUPPORTED_STACK_FORMATS = ["tif", "tiff"]


# DESIGN NOTE: tutti i path dovrebbero essere individuati in Config,
# per ogni modalità (stack, multi_stack, single_images), ogni modalità produrrà i suoi attributi 
# specifici che verranno richiesti differenziatamente nelle classi successive
# ogni problema legato a glob dei file deve essere ricondotto qui

class Config:

    def __init__(self,
                 yml_path: Path = None,
                 cfg_dict: dict = None):

        if cfg_dict is not None or yml_path is not None:
            if cfg_dict is None:
                self.yml_path = Path(yml_path)
                self.cfg_dict = {}
                self.cfg_dict = self.load_yml(yml_path)
            else:
                self.cfg_dict = cfg_dict

            self._parse_run_name()

            self.output_cfg = self.cfg_dict["output_cfg"]
            self._parse_output_cfg(self.output_cfg)

            # tiling predictor parsing
            self.tiled_predictor_cfg = self.cfg_dict["tiled_predictor_cfg"]
            self._parse_tiled_predictor_cfg(self.tiled_predictor_cfg)

            # performance evaluation parsing
            self.pe_cfg = self.get_param(self.cfg_dict, "performance_evauation_cfg", None)
            self.pe_cfg = self.cfg_dict[
                "performance_evaluation_cfg"] if "performance_evaluation_cfg" in self.cfg_dict else None
            self._parse_performance_evaluation_cfg(self.pe_cfg)

            # instance segmentation parsing
            self.is_cfg = self.get_param(self.cfg_dict, "instance_segmentation_cfg", None)
            self._parse_instance_segmentation_cfg(self.is_cfg)

            # notes parsing
            self.notes = self.cfg_dict["notes"]

    @staticmethod
    def get_param(cfg_dict: dict, key: str, default):
        """load key from dataset_cfg with default"""
        if cfg_dict is None:
            return default
        if key in cfg_dict:
            return cfg_dict[key]
        else:
            return default

    def _parse_tiled_predictor_cfg(self, tiled_predictor_cfg: dict) -> None:
        self.window_overlap = self.get_param(tiled_predictor_cfg, "window_overlap", None)
        self.extra_padding_windows = self.get_param(tiled_predictor_cfg, "extra_padding_windows", 1)
        assert type(self.extra_padding_windows) == int, "must have an integer number of extra padding windows"
        self.tiling_mode = self.get_param(tiled_predictor_cfg, "tiling_mode", "average")
        self.to_segmentation = self.get_param(tiled_predictor_cfg, "to_segmentation", False)
        self.n_tiling_threads = self.get_param(tiled_predictor_cfg, "n_threads", cpu_count())
        return

    def _parse_output_cfg(self, out_cfg: dict) -> None:
        self.output_root_path = self._decode_path(out_cfg["output_path"])
        self.output_path = self.output_root_path.joinpath(self.run_name)

        if "descriptor_path" not in out_cfg:
            self.descriptor_path = self.output_path.joinpath("descr")
        else:
            self.descriptor_path = self._decode_path(out_cfg["descriptor_path"])
        return

    # > PERFORMANCE EVALUATION PARSING <
    def _parse_performance_evaluation_cfg(self, pe_cfg: dict) -> None:
        if pe_cfg is not None:
            self.evaluate_performance = True
            self.pe_window_size = pe_cfg["window_size"]
            self.pe_batch_size = pe_cfg["batch_size"]
            # self.pe_chunk_size = pe_cfg["chunk_size"]
            self.pe_classification_threshold = self.get_param(pe_cfg, "classification_threshold", 0.5)
            self.add_empty_channel = self.get_param(pe_cfg, "add_empty_channel", False)
            self.pe_enable_curves = self.get_param(pe_cfg, "enable_curves", False)
            self.pe_multigpu = self.get_param(pe_cfg, "multi_gpu", False)
        else:
            self.evaluate_performance = False
        return

    def _parse_run_name(self) -> None:
        try:
            run_name = self.cfg_dict["run_name"]
        except KeyError:
            run_name = NameGenerator().name

        self.run_name = run_name if run_name is not None else NameGenerator().name

        return

    def _parse_instance_segmentation_cfg(self, instance_segmentation_cfg: dict) -> None:
        self.enable_instance_segmentation = self.get_param(instance_segmentation_cfg,
                                                           "enable_instance_segmentation", False)
        self.instance_segmentation_kernel_size = self.get_param(instance_segmentation_cfg,
                                                                "kernel_size", (5, 5, 5))
        self.instance_segmentation_clear_borders = self.get_param(instance_segmentation_cfg,
                                                                  "clear_borders", False)
        self.instance_segmentation_distance_transform_threshold = self.get_param(instance_segmentation_cfg,
                                                                                 "distance_transform_threshold", 0.2)
        self.instance_segmentation_distance_transform_sampling = self.get_param(instance_segmentation_cfg,
                                                                                "distance_transform_sampling", 5)
        self.instance_segmentation_watershed_line = self.get_param(instance_segmentation_cfg,
                                                                   "watershed_line", False)
        self.instance_segmentation_bg_level = self.get_param(instance_segmentation_cfg,
                                                             "bg_level", 10)
        return

    @staticmethod
    def _joinpath_mkdir(base_path: Path,
                        name: str) -> Path:
        npath = base_path.joinpath(name)
        npath.mkdir(exist_ok=True, parents=True)

        return npath

    @staticmethod
    def _decode_path(path_string: str) -> Union[Path, None]:
        """from string to pathlib Path"""
        if path_string in ["", None]:
            parsed_path = None
        else:
            parsed_path = Path(path_string)
        return parsed_path

    @staticmethod
    def _encode_path(pathlib_path: Path) -> Union[str, None]:
        """from pathlib Path to string"""
        if pathlib_path is None:
            encoded_path = None
        else:
            encoded_path = str(pathlib_path)
        return encoded_path

    @staticmethod
    def load_yml(yml_path: Path) -> dict:
        """load yml config files"""
        with yml_path.open(mode="r") as in_conf_file:
            cfg = yaml.load(in_conf_file, Loader=yaml.FullLoader)
        return cfg

    @staticmethod
    def save_yml(conf_dict: dict, output_path: Path) -> None:
        """save yml config files"""
        with output_path.open(mode="w") as out_conf_file:
            yaml.dump(conf_dict, out_conf_file)
        return

    @staticmethod
    def _is_supported_format(fpath: Path, supported_list: list) -> bool:
        """check if extension is supported"""
        extension = fpath.suffix.split(".")[1]
        return extension in supported_list


class TrainConfig(Config):
    """training configuration class"""

    def __init__(self,
                 yml_path: Path = None,
                 cfg_dict: dict = None):
        super().__init__(yml_path=yml_path, cfg_dict=cfg_dict)

        self.config_type = "training"

        self.training_cfg = self.cfg_dict["training_cfg"]
        self._parse_training_cfg(self.training_cfg)

        # optimizer cfg parsing
        self.optimizer_cfg = self.cfg_dict["optimizer_cfg"]

        # dataset cfg parsing
        self.dataset_cfg = self.cfg_dict["dataset_cfg"]
        self._parse_dataset_cfg(self.dataset_cfg)

        # model parsing
        self.model_cfg = self.cfg_dict["model_cfg"]
        self._parse_model_cfg(self.model_cfg)

        # data augmentation parsing
        self.da_cfg = self.cfg_dict["data_augmentation_cfg"]
        self._parse_data_augmentation_cfg(self.da_cfg)

        self._gen_paths(self.dataset_path)

        # callbacks parsing
        self.callbacks_cfg = self.cfg_dict["callbacks_cfg"]
        self._parse_callbacks_cfg(self.callbacks_cfg)

        # debug parsing
        self.debug_cfg = self.get_param(self.cfg_dict, "debug_cfg", {})
        self._parse_debug_cfg(self.debug_cfg)

        self._parse_performance_evaluation_cfg_additional()

    def _parse_training_cfg(self,
                            training_cfg: dict) -> None:
        """parse the training configuration dict"""
        self.training_mode = training_cfg["mode"]
        self.epochs = training_cfg["epochs"]
        self.loss = self.get_param(training_cfg, "loss", "binary_crossentropy")
        self.batch_size = self.get_param(training_cfg, "batch_size", 1)
        self.track_metrics = self.get_param(training_cfg, "track_metrics",
                                            ["accuracy", "jaccard_index", "dice_coefficient"])
        self.pos_weight = self.get_param(training_cfg, "pos_weight", None)
        self.class_weights = self.get_param(training_cfg, "class_weights", None)
        self.distribute_strategy = self.get_param(training_cfg, "distribute_strategy", None)
        self.enable_wandb_tracking = self.get_param(training_cfg, "enable_wandb_tracking", False)
        self.wandb_project = self.get_param(training_cfg, "wandb_project", None)
        self.wandb_entity = self.get_param(training_cfg, "wandb_entity", None)
        return

    def _parse_dataset_cfg(self,
                           dataset_cfg: dict) -> None:
        """parse the dataset configuration dict"""
        self.dataset_path = self._decode_path(dataset_cfg["dataset_path"])
        self.dataset_mode = dataset_cfg["mode"]
        self.n_channels = dataset_cfg["n_channels"]
        self.soft_labels = self.get_param(dataset_cfg, "soft_labels", False)
        # TODO: deprecate positive_class_vaue and negative_class_value
        self.positive_class_value = self.get_param(dataset_cfg, "positive_class_value", 255)
        self.negative_class_value = self.get_param(dataset_cfg, "negative_class_value", 0)
        self.ignore_last_channel = self.get_param(dataset_cfg, "ignore_last_channel", False)
        self.normalize_inputs = self.get_param(dataset_cfg, "normalize_inputs", True)
        self.normalize_masks = self.get_param(dataset_cfg, "normalize_masks", False)
        self.use_bboxes = self.get_param(dataset_cfg, "use_bboxes", False)

        # TODO: deprecate n_output_classes (inferred from class_values)
        #        self.class_values = self.get_param(dataset_cfg, "class_values", [1, ])
        #        self.background_value = self.get_param(dataset_cfg, "background_value", 255)
        #        self.n_output_classes = len(self.class_values)
        # TODO: n_output_classes is not used anymore
        self.n_output_classes = 1
        return

    def _parse_model_cfg(self,
                         model_cfg: dict) -> None:
        """parse model configuration dict"""
        if self.training_mode == "2d":
            if model_cfg["model"] == "resunet2d":
                self._parse_unet2d_cfg(model_cfg)
            else:
                raise NotImplementedError(model_cfg["model"])
        elif self.training_mode == "3d":
            if model_cfg["model"] == "resunet3d":
                self._parse_unet3d_cfg(model_cfg)
            else:
                raise NotImplementedError(model_cfg["model"])
        else:
            raise NotImplementedError(self.training_mode)
        return

    def _parse_unet_base(self,
                         model_cfg: dict) -> None:
        """parse unet configuration dict"""

        self.model = model_cfg["model"]
        self.window_size = model_cfg["window_size"]
        self.unet_depth = model_cfg["unet_depth"]
        self.base_filters = model_cfg["base_filters"]
        self.batch_normalization = model_cfg["batch_normalization"]
        self.residual_preactivation = model_cfg["residual_preactivation"]
        self.transposed_convolution = model_cfg["transposed_convolution"]
        return

    def _parse_unet2d_cfg(self, model_cfg: dict) -> None:
        self._parse_unet_base(model_cfg)

    def _parse_unet3d_cfg(self, model_cfg: dict) -> None:
        self._parse_unet_base(model_cfg)

    # > DATA AUGMENTATION PARSING <
    def _parse_data_augmentation_cfg(self, da_cfg: dict) -> None:
        if self.training_mode == "2d":
            self._parse_data_augmentation_cfg_2d(da_cfg)
        elif self.training_mode == "3d":
            self._parse_data_augmentation_cfg_3d(da_cfg)
        else:
            raise NotImplementedError(self.training_mode)
        return

    def _parse_data_augmentation_cfg_2d(self, da_cfg: dict) -> None:
        self._parse_data_augmentation_common(da_cfg)
        self.da_debug_mode = da_cfg["debug_mode"]
        self.da_buffer_size = da_cfg["buffer_size"]
        self.da_shuffle = self.get_param(da_cfg, "shuffle", False)
        self.da_seed = self.get_param(da_cfg, "seed", None)
        return

    def _parse_data_augmentation_cfg_3d(self, da_cfg: dict) -> None:
        self._parse_data_augmentation_common(da_cfg)
        self.da_shuffle = self.get_param(da_cfg, "shuffle", False)
        self.da_seed = self.get_param(da_cfg, "seed", None)
        return

    def _parse_data_augmentation_common(self, da_cfg: dict) -> None:
        self.da_single_thread = da_cfg["single_thread"]
        self.da_threads = da_cfg["threads"]
        self.da_transform_cfg = da_cfg["transform_cfg"]
        self.da_transforms = list(self.da_transform_cfg.keys()) if self.da_transform_cfg is not None else None
        return

    # > CALLBACK PARSING <
    def _parse_callbacks_cfg(self, callbacks_cfg: dict) -> None:
        self.callbacks = list(callbacks_cfg.keys())
        return

    def _parse_debug_cfg(self, debug_cfg: dict) -> None:
        self.test_datagen_inspector = self.get_param(debug_cfg, "test_datagen_inspector", False)
        self.val_datagen_inspector = self.get_param(debug_cfg, "val_datagen_inspector", False)
        self.train_datagen_inspector = self.get_param(debug_cfg, "train_datagen_inspector", False)
        self.predict_inspector = self.get_param(debug_cfg, "predict_inspector", False)
        return

    def _gen_paths(self, dataset_path: Path) -> None:
        """generate the path tree"""
        path_dict = {}
        if self.training_mode in ["2d", "3d"]:
            if self.dataset_mode in ["single_images", "multi_stack"]:
                for partition in ["train", "val", "test"]:
                    partition_path = dataset_path.joinpath(partition)
                    partition_subdir_dict = {}
                    for subdir in ["frames", "masks"]:
                        partition_subdir_dict[subdir] = partition_path.joinpath(subdir)
                    path_dict[partition] = partition_subdir_dict

                self.path_dict = path_dict
                self.train_paths = path_dict["train"]
                self.val_paths = path_dict["val"]
                self.test_paths = path_dict["test"]

            elif self.dataset_mode == "stack":
                for partition in ["train", "val", "test"]:
                    partition_path = dataset_path.joinpath(partition)
                    partition_subdir_dict = {}
                    for subdir in ["frames", "masks"]:
                        partition_subdir_dict[subdir] = partition_path.joinpath(subdir)
                    path_dict[partition] = partition_subdir_dict

                self.path_dict = path_dict
                self.train_paths = path_dict["train"]
                self.val_paths = path_dict["val"]
                self.test_paths = path_dict["test"]

            elif self.dataset_mode == "h5_dataset":
                for partition in ["train", "val", "test"]:
                    partition_path = dataset_path.joinpath(f"{partition}.h5")
                    path_dict[partition] = partition_path

                self.path_dict = path_dict
            elif self.dataset_mode == "multi_stack":
                raise NotImplementedError(self.dataset_mode)
            else:
                raise NotImplementedError(self.dataset_mode)

            # create output path structure
            self.output_path.mkdir(exist_ok=True, parents=True)
            self.temp_path = self._joinpath_mkdir(self.output_path, "tmp")
            self.logs_path = self._joinpath_mkdir(self.output_path, "logs")

            if self.enable_wandb_tracking:
                self.wandb_path = self._joinpath_mkdir(self.output_path, "wandb")

            self.logfile_path = self.logs_path.joinpath("train_log.log")
            self.model_history_path = self.output_path.joinpath("model_history.pickle")
            self.final_model_path = self.output_path.joinpath("final_model.hdf5")
            self.csv_summary_path = self.output_path.joinpath("run_summary.csv")
            self.custom_objects_path = self.output_path.joinpath("custom_objects.yml")
        else:
            raise NotImplementedError(self.training_mode)

        return

    def _parse_performance_evaluation_cfg_additional(self) -> None:
        """additional configs for performance evaluation in training mode"""
        self.ground_truth_mode = self.dataset_mode
        self.ground_truth_path = self.test_paths["masks"] if self.dataset_mode != "h5_dataset" else None
        self.ground_truth_normalize = self.get_param(self.pe_cfg, "normalize_ground_truth", True)
        self.ground_truth_soft_labels = self.get_param(self.pe_cfg, "soft_labels", self.soft_labels)
        self.padding_mode = self.get_param(self.pe_cfg, "padding_mode", "reflect")
        self.to_8bit = self.get_param(self.pe_cfg, "to_8bit", False)

        return


class PredictConfig(Config):
    def __init__(self,
                 yml_path: Path = None,
                 cfg_dict: dict = None):
        super().__init__(yml_path=yml_path, cfg_dict=cfg_dict)

        self.config_type = "predict"
        self.data_cfg = self.cfg_dict["data_cfg"]
        self._parse_input_data_cfg(self.data_cfg)

        self.prediction_cfg = self.cfg_dict["prediction_cfg"]
        self._parse_prediction_cfg(self.prediction_cfg)

        debug_cfg = self.get_param(self.cfg_dict, "debug_cfg", None)
        self.predict_inspector = self.get_param(debug_cfg, "predict_inspector", False)
        self.output_mode = self.output_cfg["output_mode"]

        self._gen_paths()

        self._parse_performance_evaluation_cfg_additional()

    def _parse_input_data_cfg(self, data_cfg: dict) -> None:
        self.data_path = self._decode_path(data_cfg["image_path"])

        self.data_mode = data_cfg["mode"]
        SUPPORTED_DATA_MODES = ["single_images", "stack", "multi_stack", "zetastitcher"]
        if self.data_mode not in SUPPORTED_DATA_MODES:
            raise NotImplementedError(self.data_mode)

        self.normalize_data = data_cfg["normalize_data"]
        self.n_channels = data_cfg["n_channels"]
        self.channel_names = self.get_param(data_cfg, "channel_names", None)
        self.ignore_last_channel = self.get_param(data_cfg, "ignore_last_channel", False)
        # self.positive_class_value = data_cfg["positive_class_value"]
        # self.negative_class_value = data_cfg["negative_class_value"]
        return

    def _parse_prediction_cfg(self, prediction_cfg: dict) -> None:
        # self.n_output_classes = prediction_cfg["n_output_classes"]
        self.model_path = self._decode_path(prediction_cfg["model_path"])
        self.custom_objects_path = self._decode_path(prediction_cfg["custom_objects_path"])
        self.prediction_mode = prediction_cfg["mode"]
        self.window_size = prediction_cfg["window_size"]
        self.batch_size = self.get_param(prediction_cfg, "batch_size", 1)
        self.padding_mode = self.get_param(prediction_cfg, "padding_mode", "reflect")
        self.n_output_classes = self.get_param(prediction_cfg, "n_output_classes", 1)
        self.class_values = self.get_param(prediction_cfg, "class_values", None)
        self.to_8bit = self.get_param(prediction_cfg, "to_8bit", False)
        self.multi_gpu = self.get_param(prediction_cfg, "multi_gpu", False)
        return

    def _gen_paths(self) -> None:
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.temp_path = self._joinpath_mkdir(self.output_path, "tmp")
        self.logs_path = self._joinpath_mkdir(self.output_path, "logs")
        return

    def _parse_performance_evaluation_cfg_additional(self) -> None:
        self.ground_truth_mode = self.get_param(self.pe_cfg, "data_mode", None)
        self.ground_truth_normalize = self.get_param(self.pe_cfg, "normalize_ground_truth", False)
        self.ground_truth_path = self._decode_path(self.get_param(self.pe_cfg, "ground_truth_path", None))
        self.ground_truth_soft_labels = self.get_param(self.pe_cfg, "soft_labels", False)

        # self.ground_truth_normalize = self.get_param(self.pe_cfg, normalize_data)
        return
