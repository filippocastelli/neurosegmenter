from pathlib import Path
import yaml
from typing import Union

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
            self.pe_cfg = self.cfg_dict[
                "performance_evaluation_cfg"] if "performance_evaluation_cfg" in self.cfg_dict else None
            self._parse_performance_evaluation_cfg(self.pe_cfg)

            # notes parsing
            self.notes = self.cfg_dict["notes"]

    def _parse_tiled_predictor_cfg(self, tiled_predictor_cfg: dict) -> None:
        self.window_overlap = tiled_predictor_cfg["window_overlap"] if "window_overlap" in tiled_predictor_cfg else None
        self.extra_padding_windows = tiled_predictor_cfg[
            "extra_padding_windows"] if "extra_padding_windows" in tiled_predictor_cfg else 1
        assert type(self.extra_padding_windows) == int, "must have an integer number of extra padding windows"
        self.tiling_mode = tiled_predictor_cfg["tiling_mode"] if "tiling_mode" in tiled_predictor_cfg else "average"
        return

    def _parse_output_cfg(self, out_cfg: dict) -> None:
        self.output_root_path = self._decode_path(out_cfg["output_path"])
        self.output_path = self.output_root_path.joinpath(self.run_name)
        self.descriptor_path = self._decode_path(out_cfg["descriptor_path"])
        self.enable_wandb_tracking = out_cfg["enable_wandb_tracking"]
        return

    # > PERFORMANCE EVALUATION PARSING <
    def _parse_performance_evaluation_cfg(self, pe_cfg: dict) -> None:
        if pe_cfg is not None:
            self.evaluate_performance = True
            self.pe_window_size = pe_cfg["window_size"]
            self.pe_batch_size = pe_cfg["batch_size"]
            # self.pe_chunk_size = pe_cfg["chunk_size"]
            self.pe_classification_threshold = pe_cfg["classification_threshold"]
            self.pe_add_empty_channel = pe_cfg["add_empty_channel"]
            self.pe_enable_curves = pe_cfg["enable_curves"] if "enable_curves" in pe_cfg else False
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

    @staticmethod
    def _joinpath_mkdir(base_path: Path,
                        name: str) -> Path:
        npath = base_path.joinpath(name)
        npath.mkdir(exist_ok=True, parents=True)

        return npath

    @staticmethod
    def _decode_path(path_string: str) -> Union[Path, None]:
        """from string to pathlib Path"""
        if path_string == "":
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
        self.debug_cfg = self.cfg_dict["debug_cfg"]
        self._parse_debug_cfg(self.debug_cfg)

        self._parse_performance_evaluation_cfg_additional()

    def _parse_training_cfg(self,
                            training_cfg: dict) -> None:
        """parse the training configuration dict"""
        self.training_mode = training_cfg["mode"]
        self.epochs = training_cfg["epochs"]
        self.loss = training_cfg["loss"]
        self.batch_size = training_cfg["batch_size"]
        self.track_metrics = training_cfg["track_metrics"]
        self.pos_weight = training_cfg["pos_weight"] if "pos_weight" in training_cfg else None
        return

    def _parse_dataset_cfg(self,
                           dataset_cfg: dict) -> None:
        """parse the dataset configuration dict"""
        self.dataset_path = self._decode_path(dataset_cfg["dataset_path"])
        self.dataset_mode = dataset_cfg["mode"]
        self.n_channels = dataset_cfg["n_channels"]
        self.positive_class_value = dataset_cfg["positive_class_value"]
        self.negative_class_value = dataset_cfg["negative_class_value"]
        self.ignore_last_channel = dataset_cfg["ignore_last_channel"] if "ignore_last_channel" in dataset_cfg else False
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
        self.crop_shape = model_cfg["crop_shape"]
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
        return

    def _parse_data_augmentation_cfg_3d(self, da_cfg: dict) -> None:
        self._parse_data_augmentation_common(da_cfg)
        self.da_shuffle = da_cfg["shuffle"]
        if "seed" in da_cfg:
            self.da_seed = da_cfg["seed"]
        else:
            self.da_seed = None
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
        default_debug_cfg = {
            "test_datagen_inspector": False,
            "train_datagen_inspector": False,
            "val_datagen_inspector": False
        }
        default_debug_cfg.update(debug_cfg)
        self.test_datagen_inspector = default_debug_cfg["test_datagen_inspector"]
        self.train_datagen_inspector = default_debug_cfg["train_datagen_inspector"]
        self.val_datagen_inspector = default_debug_cfg["val_datagen_inspector"]
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
        else:
            raise NotImplementedError(self.training_mode)

        return

    def _parse_performance_evaluation_cfg_additional(self) -> None:
        """additional configs for performance evaluation in training mode"""
        self.ground_truth_mode = self.dataset_mode
        self.ground_truth_path = self.test_paths["masks"]
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

        self.output_mode = self.output_cfg["output_mode"]

        self._gen_paths()

        self._parse_performance_evaluation_cfg_additional()

    def _parse_input_data_cfg(self, data_cfg: dict) -> None:
        self.data_path = self._decode_path(data_cfg["image_path"])

        self.data_mode = data_cfg["mode"]
        SUPPORTED_DATA_MODES = ["single_images", "stack", "multi_stack"]
        if self.data_mode not in SUPPORTED_DATA_MODES:
            raise NotImplementedError(self.data_mode)

        self.normalize_data = data_cfg["normalize_data"]
        self.n_channels = data_cfg["n_channels"]
        # self.positive_class_value = data_cfg["positive_class_value"]
        # self.negative_class_value = data_cfg["negative_class_value"]
        return

    def _parse_prediction_cfg(self, prediction_cfg: dict) -> None:
        self.n_output_classes = prediction_cfg["n_output_classes"]
        self.model_path = self._decode_path(prediction_cfg["model_path"])
        self.prediction_mode = prediction_cfg["mode"]
        self.window_size = prediction_cfg["window_size"]
        self.batch_size = prediction_cfg["batch_size"]
        self.padding_mode = prediction_cfg["padding_mode"] if "padding_mode" in prediction_cfg else "reflect"
        return

    def _gen_paths(self) -> None:
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.temp_path = self._joinpath_mkdir(self.output_path, "tmp")
        self.logs_path = self._joinpath_mkdir(self.output_path, "logs")
        return

    def _parse_performance_evaluation_cfg_additional(self) -> None:
        self.ground_truth_mode = self.pe_cfg["data_mode"]
        self.ground_truth_path = self._decode_path(self.pe_cfg["ground_truth_path"])
        return