import shutil
import pickle
import os
import zipfile
from pathlib import Path
import logging
import yaml


class RunDescriptorLight:
    def __init__(self,
                 config=None,
                 load_from_archive=None,
                 performance_metrics_dict=None):
        
        self.load_from_archive = load_from_archive
        
        self.performance_metrics_dict = performance_metrics_dict
        self.config = config
        self.config_type = config.config_type
        self._dump_descriptor()
        
    def _dump_descriptor(self):
        self._make_paths()
        self._copy_model()
        self._copy_config()
        self._copy_logs()
        self._serialize_performance_dict()
        self._create_archive()
        
    def _make_paths(self):
        """create main paths"""
        self.yml_path = self.config.yml_path
        self.descriptor_path = self.config.descriptor_path
        self.descriptor_local_path = self.descriptor_path.joinpath("tmp")        
        self.descriptor_local_path.mkdir(parents=True, exist_ok=True)
        self.original_logs_path = self.config.logs_path
    
    def _copy_model(self):
        """copy final model"""
        if self.config_type == "training":
            self.final_model_original_path = self.config.final_model_path
            self.final_model_local_path = self.descriptor_òpcaò_path.joinpath("logs")
            
            shutil.copy(str(self.final_model_original_path), str(self.final_model_local_path))
    
    def _copy_logs(self):
        """copy logs path"""
        self.log_dir_local_path = self.descriptor_local_path.joinpath("logs")
        shutil.copytree(str(self.original_logs_path), str(self.log_dir_local_path))
        
    def _copy_config(self):
        """copy yml config""" 
        self.yml_local_path = self.descriptor_local_path.joinpath("config.yml")
        shutil.copy(str(self.yml_path), str(self.yml_local_path))

    def _serialize_performance_dict(self):
        """serialize model performance metrics as a pickle"""
        if self.performance_metrics_dict is not None:
            self.performance_metrics_local_pickle_path = self.descriptor_local_path.joinpath("performance.pickle")
            self._to_pickle(self.performance_metrics_dict, self.performance_metrics_local_pickle_path)
    
    @staticmethod
    def _to_pickle(obj, fpath):
        """Save obj to picklable file in fpath"""
        with fpath.open(mode="wb") as pickle_out:
            pickle.dump(obj, pickle_out)
            
    def _create_archive(self):
        self.archive_path = self.descriptor_path.joinpath("archive.zip")
        
        previous_path = Path.cwd()
        os.chdir(str(self.descriptor_local_path))
        
        with zipfile.ZipFile(str(self.archive_path), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for fpath in self.descriptor_local_path.rglob("*"):
                relative_path = fpath.relative_to(self.descriptor_local_path)
                zf.write(str(relative_path))
                
        zf.close()
        os.chdir(str(previous_path))
        
    def load_from_archive(self, archive_path=None):
        archive_path = self._string_to_path(archive_path)
        
        archive = zipfile.ZipFile(str(archive_path))
        
        if archive.testzip() is not None:
            logging.warning(
                "archive may be corrupted, found invalid files {}".format(
                    archive.testzip()))
        
        yml_files = [fname for fname in archive.namelist() if "yml" in fname]
        
        assert len(yml_files) > 0, "Can't find .yml config in archive {}".format(archive_path)
        
        yml_file = archive.read(yml_files[0])
        yml_dict = yaml.load(yml_file,Loader=yaml.FullLoader)
        
        
        # extract archive
        
        # populate config
        
        
    @staticmethod
    def _string_to_path(in_string):
        return Path(in_string) if in_string is not None else None
