import pytest
from mock import patch

from pathlib import Path
import os, sys, inspect
# import pudb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import yaml


from utils import load_volume
# from config import TrainConfig

CURRENT_DIR_PATH = Path(currentdir)
YML_CONFIG_PATH = CURRENT_DIR_PATH.joinpath("test_cfg_datagen.yml")
TRAIN_CFG_PATH = CURRENT_DIR_PATH.joinpath("test_train_cfg.yml")


class YMLio:
    @staticmethod
    def read_yml(yml_path):
        with yml_path.open(mode="r") as yml_in:
            dict_yml = yaml.load(yml_in, Loader=yaml.FullLoader)
        return dict_yml
    
    @staticmethod
    def write_yml(yml_dict, yml_out_path):
        with yml_out_path.open(mode="w") as yml_out:
            yaml.dump(yml_dict, yml_out)
        return
    
    @classmethod
    def combine_ymls(cls, yml_dict_default_path, yml_dict_path, yml_out_path):
        yml_dict = cls.read_yml(yml_dict_path)
        yml_default_dict = cls.read_yml(yml_dict_default_path)
        yml_default_dict.update(yml_dict)
        cls.write_yml(yml_default_dict, yml_out_path)
        return 
    
    
@pytest.fixture(scope="module", params=["single_images", "stack", "multi_stack"])
def config_yml_path_fixture(request):
    dataset_yml_name = f"dataset_{request.param}.yml"
    dataset_yml_path = CURRENT_DIR_PATH.joinpath(dataset_yml_name)
    # print(dataset_yml_name)
    default_yml_path = TRAIN_CFG_PATH
    YMLio.combine_ymls(yml_dict_default_path=default_yml_path,
                        yml_dict_path=dataset_yml_path,
                        yml_out_path=YML_CONFIG_PATH)
        
    return YML_CONFIG_PATH, request.param


class TestUtilsVolumeIO:
    
    @pytest.mark.utils
    def test_load_volume(self):
        pass
    

        

    

