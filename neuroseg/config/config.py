from pathlib import Path
import yaml


class TrainConfig:
    
    def __init__(self,
                 yml_path):
        
        self.yml_path = Path(yml_path)
        self.cfg_dict = {}
        self.load(self.yml_path)
        
    def load(self, yml_path):
        cfg_dict = self.load_yml(yml_path)
        
        #Global vars
        self.training_mode = cfg_dict["training_mode"]
        self.dataset_path = self._decode_path(cfg_dict["dataset_path"])
        self.output_path = self._decode_path(cfg_dict["output_path"])
        self.descriptor_path = self._decode_path(cfg_dict["descriptor_path"])
        
        self.epochs = cfg_dict["epochs"]
        self.batch_size = cfg_dict["batch_size"]
        self.initial_learning_rate = cfg_dict["initial_learning_rate"]
        self.loss = cfg_dict["loss"]
        self.track_metrics = cfg_dict["track_metrics"]
        self.enable_wandb_tracking = cfg_dict["enable_wandb_tracking"]
        
        # path dict
        self._gen_paths(self.dataset_path)
        
        # dataset cfg parsing
        self.dataset_cfg = cfg_dict["dataset_cfg"]
        self._parse_dataset_cfg(self.dataset_cfg)
        
        # model parsing
        self.model_cfg = cfg_dict["model_cfg"]
        self._parse_model_cfg(self.model_cfg)
        
        # data augmentation parsing
        self.da_cfg = cfg_dict["data_augmentation_cfg"]
        self._parse_data_augmentation_cfg(self.da_cfg)
        
        # performance evaluation parsing
        self.pe_cfg = cfg_dict["performance_evaluation_cfg"]
        self._parse_performance_evaluation_cfg(self.pe_cfg)
        
        # callbacks parsing
        self.callbacks_cfg = cfg_dict["callbacks_cfg"]
        self._parse_callbacks_cfg(self.callbacks_cfg)
        
        # notes parsing
        self.notes = cfg_dict["notes"]
    
    # > DATASET PARSING
    def _parse_dataset_cfg(self, dataset_cfg):
        self.dataset_mode = dataset_cfg["mode"]
        self.positive_class_value = dataset_cfg["positive_class_value"]
        self.negative_class_value = dataset_cfg["negative_class_value"]
    
    # > MODEL PARSING <
    def _parse_model_cfg(self, model_cfg):
        if self.training_mode == "2d":
            if model_cfg["model"] == "unet2d":
                self._parse_unet2d_cfg(model_cfg)
            elif model_cfg["model"] == "unet3d":
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
        else:
            raise NotImplementedError(self.training_mode)
    
    def _parse_data_augmentation_cfg_2d(self, da_cfg):
        self.da_debug_mode = da_cfg["debug_mode"]
        self.da_single_thread = da_cfg["single_thread"]
        self.da_threads = da_cfg["threads"]
        self.da_buffer_size = da_cfg["buffer_size"]
        
        self.da_transform_cfg = da_cfg["transform_cfg"]
        self.da_transforms = list(self.da_transform_cfg.keys()) if self.da_transform_cfg is not None else None
        
    # > PERFORMANCE EVALUATION PARSING <
    def _parse_performance_evaluation_cfg(self, pe_cfg):
        self.pe_batch_size = pe_cfg["batch_size"]
        self.pe_chunk_size = pe_cfg["chunk_size"]
        self.pe_classification_threashold = pe_cfg["classification_threshold"]
        self.pe_add_empty_channel = pe_cfg["add_empty_channel"]
        
        
    # > CALLBACK PARSING <
    def _parse_callbacks_cfg(self, callbacks_cfg):
        self.callbacks = list(callbacks_cfg.keys())
        
        
    def _gen_paths(self, dataset_path):
        
        path_dict = {}
        if self.training_mode == "2d":
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
                
            # create output path structure
            self.output_path.mkdir(exist_ok=True, parents=True)
            self.temp_path = self._joinpath_mkdir(self.output_path, "tmp")
            self.logs_path = self._joinpath_mkdir(self.output_path, "tmp")

            if self.enable_wandb_tracking:
                self.wandb_path = self._joinpath_mkdir(self.output_path, "wandb")
                
            self.logfile_path = self.logs_path.joinpath("train_log.log")
            self.model_history_path = self.output_path.joinpath("model_history.pickle")
            self.final_model_path = self.output_path.joinpath("final_model.hdf5")
            self.csv_summary_path = self.output_path.joinpath("run_summary.csv")
        else:
            raise NotImplementedError(self.training_mode)
            
        
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
    
    
if __name__ == "__main__":
    import platform
    
    if platform.system() == "Windows":
        cfg_path = Path("C:/Users/filip/Documenti/GitHub/neuroseg/neuroseg/config/ex_cfg.yml")
    else:
        cfg_path = Path("/home/phil/repos/neuroseg/neuroseg/config/ex_cfg.yml")
    config = TrainConfig(cfg_path)