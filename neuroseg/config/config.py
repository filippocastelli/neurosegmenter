from pathlib import Path
import yaml


class TrainingConfig:
    
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
        self.epochs = cfg_dict["epochs"]
        self.batchsize = cfg_dict["batchsize"]
        self.initial_learning_rate = cfg_dict["initial_learning_rate"]
        self.loss = cfg_dict["loss"]
        self.track_metrics = cfg_dict["track_metrics"]
        
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
        
        self.da_transform_cfg = da_cfg["transform_cfg"]
        self.da_transforms = list(self.da_transform_cfg.keys())
        
    # > PERFORMANCE EVALUATION PARSING <
    def _parse_performance_evaluation_cfg(self, pe_cfg):
        self.pe_batchsize = pe_cfg["batchsize"]
        self.pe_chunksize = pe_cfg["chunksize"]
        self.pe_classification_threashold = pe_cfg["classification_threshold"]
        self.pe_add_empty_channel = pe_cfg["add_empty_channel"]
        
        
    # > CALLBACK PARSING <
    def _parse_callbacks_cfg(self, callbacks_cfg):
        self.callbacks = list(callbacks_cfg.keys())
        
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
    config = TrainingConfig(cfg_path)