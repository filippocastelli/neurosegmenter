from pathlib import Path
import yaml

from neuroseg.utils import NameGenerator

SUPPORTED_STACK_FORMATS = ["tif", "tiff"]

# DESIGN NOTE: tutti i path dovrebbero essere individuati in Config, 
# per ogni modalità (stack, multi_stack, single_images), ogni modalità produrrà i suoi attributi 
# specifici che verranno richiesti differenziatamente nelle classi successive
# ogni problema legato a glob dei file deve essere ricondotto qui

class Config:
    
    def __init__(self,
                 yml_path):
        
        self.yml_path = Path(yml_path)
        self.cfg_dict = {}
        self.cfg_dict = self.load_yml(yml_path)
        self._parse_run_name()
        
        self.output_cfg = self.cfg_dict["output_cfg"]
        self._parse_output_cfg(self.output_cfg)
        
        # performance evaluation parsing
        self.pe_cfg = self.cfg_dict["performance_evaluation_cfg"] if "performance_evaluation_cfg" in self.cfg_dict else None
        self._parse_performance_evaluation_cfg(self.pe_cfg)
        
        # notes parsing
        self.notes = self.cfg_dict["notes"]
        
        
    def _parse_output_cfg(self, out_cfg):
        self.output_root_path = self._decode_path(out_cfg["output_path"])
        self.output_path = self.output_root_path.joinpath(self.run_name)
        self.descriptor_path = self._decode_path(out_cfg["descriptor_path"])
        self.enable_wandb_tracking = out_cfg["enable_wandb_tracking"]
            
    # > PERFORMANCE EVALUATION PARSING <
    def _parse_performance_evaluation_cfg(self, pe_cfg):
        if pe_cfg is not None:
            self.evaluate_performance = True
            self.pe_window_size = pe_cfg["window_size"]
            self.pe_batch_size = pe_cfg["batch_size"]
            self.pe_chunk_size = pe_cfg["chunk_size"]
            self.pe_classification_threshold = pe_cfg["classification_threshold"]
            self.pe_add_empty_channel = pe_cfg["add_empty_channel"]
            self.pe_enable_curves = pe_cfg["enable_curves"] if "enable_curves" in pe_cfg else False
        else:
            self.evaluate_performance = False
            
    def _parse_run_name(self):
        try:
            run_name = self.cfg_dict["run_name"]
        except KeyError:
            run_name = NameGenerator().name
        
        self.run_name = run_name if run_name is not None else NameGenerator().name
        
    @staticmethod
    def _joinpath_mkdir(base_path, name):
        npath = base_path.joinpath(name)
        npath.mkdir(exist_ok=True, parents=True)
        
        return npath
        
    def save(self):
        pass
    
    @staticmethod
    def _decode_path(path_string):
        """from string to pathlib Path""" 
        if path_string == "":
            parsed_path = None
        else:
            parsed_path = Path(path_string)
        return parsed_path

    @staticmethod
    def _encode_path(pathlib_path):
        """from pathlib Path to string"""
        if pathlib_path is None:
            encoded_path = None
        else:
            encoded_path = str(pathlib_path)
        return encoded_path
    
    @staticmethod
    def load_yml(yml_path):
        """load yml config files"""
        with yml_path.open(mode="r") as in_conf_file:
            cfg = yaml.load(in_conf_file, Loader=yaml.FullLoader)
        return cfg
    
    @staticmethod
    def save_yml(conf_dict, output_path):
        """save yml config files"""
        with output_path.open(mode="w") as out_conf_file:
            yaml.dump(conf_dict, out_conf_file)
        return
    
    @staticmethod
    def _is_supported_format(fpath, supported_list):
        extension = fpath.suffix.split(".")[1]
        return extension in supported_list
    
class TrainConfig(Config):
    def __init__(self, yml_path):
        super().__init__(yml_path)
        
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
        
    def _parse_training_cfg(self, training_cfg):
        self.training_mode = training_cfg["mode"]
        self.epochs = training_cfg["epochs"]
        self.loss = training_cfg["loss"]
        self.batch_size = training_cfg["batch_size"]
        self.track_metrics = training_cfg["track_metrics"]
        
    def _parse_dataset_cfg(self, dataset_cfg):
        self.dataset_path = self._decode_path(dataset_cfg["dataset_path"])
        self.dataset_mode = dataset_cfg["mode"]
        self.n_channels = dataset_cfg["n_channels"]
        self.positive_class_value = dataset_cfg["positive_class_value"]
        self.negative_class_value = dataset_cfg["negative_class_value"]
        
        self.ignore_last_channel = dataset_cfg["ignore_last_channel"] if "ignore_last_channel" in dataset_cfg else False
    
    # > MODEL PARSING <
    def _parse_model_cfg(self, model_cfg):
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
              
    def _parse_unet_base(self, model_cfg):
        self.model = model_cfg["model"]
        self.crop_shape = model_cfg["crop_shape"]
        self.unet_depth = model_cfg["unet_depth"]
        self.base_filters = model_cfg["base_filters"]
        self.batch_normalization = model_cfg["batch_normalization"]
        self.residual_preactivation = model_cfg["residual_preactivation"]
        self.transposed_convolution = model_cfg["transposed_convolution"]
   
    def _parse_unet2d_cfg(self, model_cfg):
        self._parse_unet_base(model_cfg)
        
    def _parse_unet3d_cfg(self, model_cfg):
        self._parse_unet_base(model_cfg)
        
    # > DATA AUGMENTATION PARSING <
    def _parse_data_augmentation_cfg(self, da_cfg):
        if self.training_mode == "2d":
            self._parse_data_augmentation_cfg_2d(da_cfg)
        elif self.training_mode == "3d":
            self._parse_data_augmentation_cfg_3d(da_cfg)
        else:
            raise NotImplementedError(self.training_mode)
    
    def _parse_data_augmentation_cfg_2d(self, da_cfg):
        self._parse_data_augmentation_common(da_cfg)
        self.da_debug_mode = da_cfg["debug_mode"]
        self.da_buffer_size = da_cfg["buffer_size"]
    
    def _parse_data_augmentation_cfg_3d(self, da_cfg):
        self._parse_data_augmentation_common(da_cfg)
        self.da_shuffle = da_cfg["shuffle"]
        if "seed" in da_cfg:
            self.da_seed = da_cfg["seed"]
        else:
            self.da_seed = None
            
        # if "pre_crop_scales" in da_cfg:
        #     self.da_pre_crop_scales = da_cfg["pre_crop_scales"]
        # else:
        #     self.da_pre_crop_scales = None
        
    def _parse_data_augmentation_common(self, da_cfg):
        self.da_single_thread = da_cfg["single_thread"]
        self.da_threads = da_cfg["threads"]
        self.da_transform_cfg = da_cfg["transform_cfg"]
        self.da_transforms = list(self.da_transform_cfg.keys()) if self.da_transform_cfg is not None else None
        
    # > CALLBACK PARSING <
    def _parse_callbacks_cfg(self, callbacks_cfg):
        self.callbacks = list(callbacks_cfg.keys())
        
    def _parse_debug_cfg(self, debug_cfg):
        default_debug_cfg = {
            "test_datagen_inspector": False,
            "train_datagen_inspector": False,
            "val_datagen_inspector": False
            }
        default_debug_cfg.update(debug_cfg)
        self.test_datagen_inspector = default_debug_cfg["test_datagen_inspector"]
        self.train_datagen_inspector = default_debug_cfg["train_datagen_inspector"]
        self.val_datagen_inspector = default_debug_cfg["val_datagen_inspector"]     
        
    def _gen_paths(self, dataset_path):
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
                # I FILE ANDREBBERO GLOBBATI QUI
                
            elif self.dataset_mode == "stack":
                # stack_file_dict = {"frames": "frames.tif",
                #                    "masks": "masks.tif"}
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
            
    def _parse_performance_evaluation_cfg_additional(self):        
        self.ground_truth_mode = self.dataset_mode
        self.ground_truth_path  = self.test_paths["masks"]

            
            
class PredictConfig(Config):
    def __init__(self, yml_path):
        super().__init__(yml_path)
        
        self.config_type = "predict"
        self.data_cfg = self.cfg_dict["data_cfg"]
        self._parse_input_data_cfg(self.data_cfg)
        
        self.prediction_cfg = self.cfg_dict["prediction_cfg"]
        self._parse_prediction_cfg(self.prediction_cfg)
        
        self.output_mode = self.output_cfg["output_mode"]
        
        self._gen_paths()
        
        self._parse_performance_evaluation_cfg_additional()
    
    def _parse_input_data_cfg(self, data_cfg):
        self.data_path = self._decode_path(data_cfg["image_path"])
        SUPPORTED_DATA_MODES = ["single_images", "stack", "multi_stack"]
        self.data_mode = data_cfg["mode"]
        self.normalize_data = data_cfg["normalize_data"]
        
        if self.data_mode not in SUPPORTED_DATA_MODES:
            raise NotImplementedError(self.data_mode)
        
        self.n_channels = data_cfg["n_channels"]
        self.positive_class_value = data_cfg["positive_class_value"]
        self.negative_class_value = data_cfg["negative_class_value"]
        
    def _parse_prediction_cfg(self, prediction_cfg):
        self.n_output_classes = prediction_cfg["n_output_classes"]
        self.model_path = self._decode_path(prediction_cfg["model_path"])
        self.prediction_mode = prediction_cfg["mode"]
        self.window_size = prediction_cfg["window_size"]
        self.batch_size = prediction_cfg["batch_size"]
        self.chunk_size = prediction_cfg["chunk_size"]
        self.padding_mode = prediction_cfg["padding_mode"]
        self.keep_tmp = prediction_cfg["keep_tmp"]
        
    def _gen_paths(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.temp_path = self._joinpath_mkdir(self.output_path, "tmp")
        self.logs_path = self._joinpath_mkdir(self.output_path, "logs")
        
    def _parse_performance_evaluation_cfg_additional(self):
        self.ground_truth_mode = self.pe_cfg["data_mode"]
        self.ground_truth_path = self._decode_path(self.pe_cfg["ground_truth_path"])
        
if __name__ == "__main__":
    import platform
    
    if platform.system() == "Windows":
        cfg_path = Path("C:/Users/filip/Documenti/GitHub/neuroseg/neuroseg/config/ex_cfg.yml")
    else:
        cfg_path = Path("/home/phil/repos/neuroseg/neuroseg/config/ex_cfg.yml")
    config = TrainConfig(cfg_path)