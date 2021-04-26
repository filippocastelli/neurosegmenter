import pytest
from mock import patch

from pathlib import Path
import shutil
import os, sys, inspect
import pudb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import tifffile
import yaml
import numpy as np

from utils import load_volume
# from config import TrainConfig

CURRENT_DIR_PATH = Path(currentdir)

TMP_TEST_PATH = CURRENT_DIR_PATH.joinpath("tmp_tests")
TMP_IMG_PATH = TMP_TEST_PATH.joinpath("test_vol_io.tiff")
    
    
YML_CONFIG_PATH = CURRENT_DIR_PATH.joinpath("test_cfg_datagen.yml")
TRAIN_CFG_PATH = CURRENT_DIR_PATH.joinpath("test_train_cfg.yml")

DATA_MODES = ["stack", "single_images", "multi_stack"]

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

@pytest.fixture(scope="module",params=["single_images", "stack", "multi_stack"])
def get_img_fixture(request):
    TMP_TEST_PATH.mkdir(exist_ok=True)
    def _get_img_fixture(channels):
        data_mode = request.param
        if data_mode == "stack":
            # pudb.set_trace()
            img_shape = (10,512,512,channels)
            img = np.random.randint(low=0, high=255, size=img_shape, dtype=np.uint8)
            tifffile.imwrite(str(TMP_IMG_PATH), img)
            return TMP_IMG_PATH, img_shape
        elif data_mode == "single_images":
            single_images_paths = []
            img_shape = (512, 512, channels)
            single_img_dir = TMP_TEST_PATH.joinpath("single_images_tmp")
            single_img_dir.mkdir(exist_ok=True)
            for idx in range(10):
                img = np.random.randint(low=0, high=255, size=img_shape, dtype=np.uint8)
                single_img_path = single_img_dir.joinpath(f"single_images_vol_io_{idx}.tiff")
                tifffile.imwrite(str(single_img_path), img)
                single_images_paths.append(single_img_path)
            return single_images_paths, img_shape
        elif data_mode == "multi_stack":
            img_shape = (10, 512, 512, channels)
            multi_stack_dir = TMP_TEST_PATH.joinpath("multi_stack_tmp")
            multi_stack_dir.mkdir(exist_ok=True)
            multi_stack_paths = []
            for idx in range(2):
                img = np.random.randint(low=0, high=255, size=img_shape, dtype=np.uint8)
                img_stack_path = multi_stack_dir.joinpath(f"multi_stack_vol_io_{idx}.tiff")
                tifffile.imwrite(str(img_stack_path), img)
                multi_stack_paths.append(img_stack_path)
            return multi_stack_paths, img_shape
        else:
            raise ValueError("unsupported mode {}".format(data_mode))
    yield _get_img_fixture, request.param
    shutil.rmtree(TMP_TEST_PATH)
    

class TestUtilsVolumeIO:
    @pytest.mark.neuroseg_utils
    @pytest.mark.parametrize("channels", [1,2,3,4], ids=["1ch", "2ch", "3ch", "4ch"])
    def test_load_volume(self, get_img_fixture, channels):
        # pudb.set_trace()
        img_fixture_factory, data_mode = get_img_fixture
        img_path, img_shape = img_fixture_factory(channels)
        
        vol = load_volume(imgpath=img_path,
                          data_mode=data_mode)
        
        if data_mode in ["stack", "single_images"]:
            # pudb.set_trace()
            pass
        elif data_mode == "multi_stack":
            pass
        else:
            raise ValueError(f"unsupported data_mode {data_mode}")