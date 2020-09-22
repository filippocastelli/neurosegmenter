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
    
    cfg_path = Path("/home/phil/repos/neuroseg/neuroseg/config/ex_cfg.yml")
    config = TrainingConfig(cfg_path)