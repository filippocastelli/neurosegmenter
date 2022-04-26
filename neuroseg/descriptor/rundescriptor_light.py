import shutil
import pickle
import os
import zipfile
from pathlib import Path
import logging
import yaml
from typing import Union
from neuroseg.config import TrainConfig, PredictConfig


class RunDescriptorLight:
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig] = None,
                 load_from_archive: bool = None,
                 performance_metrics_dict: dict = None,
                 model_history_dict: dict = None):

        self.load_from_archive = load_from_archive

        self.performance_metrics_dict = performance_metrics_dict
        self.model_history_dict = model_history_dict
        self.config = config
        self.run_name = config.run_name
        self.config_type = config.config_type
        self._dump_descriptor()

    @staticmethod
    def beautify_dict(dictionary: dict):
        dict_copy = dictionary.copy()
        for key, value in dict_copy.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    value[subkey] = str(subval)
            else:
                dict_copy[key] = str(value)
        return dict_copy

    def _dump_descriptor(self) -> None:
        self._make_paths()
        self._copy_model()
        self._copy_config()
        self._copy_logs()
        self._serialize_performance_dict()
        self._create_archive()

    def _make_paths(self) -> None:
        """create main paths"""
        self.yml_path = self.config.yml_path
        self.descriptor_path = self.config.descriptor_path
        self.descriptor_local_path = self.descriptor_path.joinpath(self.run_name)
        self.descriptor_local_path.mkdir(parents=True, exist_ok=True)
        self.original_logs_path = self.config.logs_path

    def _copy_model(self) -> None:
        """copy final model"""
        if self.config_type == "training":
            self.final_model_original_path = self.config.final_model_path
            self.final_model_local_path = self.descriptor_local_path.joinpath("final_model.hdf5")

            shutil.copyfile(str(self.final_model_original_path), str(self.final_model_local_path))

    def _copy_logs(self) -> None:
        """copy logs path"""
        self.log_dir_local_path = self.descriptor_local_path.joinpath("logs")
        self._copydir(source_dir=self.original_logs_path, dest_dir=self.log_dir_local_path)
        # shutil.copytree(str(self.original_logs_path), str(self.log_dir_local_path), copy_function=shutil.copyfile)
        # python 3.6 is not compatible with dirs_exist_ok argument, only 3.8
        # shutil.copytree(str(self.original_logs_path), str(self.log_dir_local_path), dirs_exist_ok=True)

    def _copy_config(self) -> None:
        """copy yml config"""
        self.yml_local_path = self.descriptor_local_path.joinpath("config.yml")
        shutil.copyfile(str(self.yml_path), str(self.yml_local_path))

    def _serialize_performance_dict(self) -> None:
        """serialize model performance metrics as a pickle"""
        if self.performance_metrics_dict is not None:
            self.performance_metrics_local_pickle_path = self.descriptor_local_path.joinpath("performance.pickle")
            self._to_pickle(self.performance_metrics_dict, self.performance_metrics_local_pickle_path)
            self.performance_metrics_local_yml_path = self.descriptor_local_path.joinpath("performance.yml")
            self._to_yml(self.beautify_dict(self.performance_metrics_dict), self.performance_metrics_local_yml_path)

    def _serialize_metrics_dict(self) -> None:
        """serialize model history"""
        if self.model_history_dict is not None:
            self.model_history_local_pickle_path = self.descriptor_local_path.joinpath("model_history.pickle")
            self._to_pickle(self.model_history_dict, self.model_history_local_pickle_path)
            self.model_history_local_yml_path = self.descriptor_local_path.joinpath("model_history.yml")
            self._to_yml(self.beautify_dict(self.model_history_dict.__dict__), self.model_history_local_yml_path)

    @staticmethod
    def _copydir(source_dir: Path,
                 dest_dir: Path):
        for fpath in source_dir.rglob("*"):
            relative_parent = fpath.relative_to(source_dir).parent
            out_dir_path = dest_dir.joinpath(relative_parent)
            out_dir_path.mkdir(exist_ok=True, parents=True)
            out_path = out_dir_path.joinpath(fpath.name)
            shutil.copyfile(str(fpath), str(out_path))

    @staticmethod
    def _to_pickle(obj, fpath: Path) -> None:
        """Save obj to picklable file in fpath"""
        with fpath.open(mode="wb") as pickle_out:
            pickle.dump(obj, pickle_out)

    @staticmethod
    def _to_yml(dictionary: dict,
                fpath: Path) -> None:
        with fpath.open(mode="w") as outfile:
            yaml.dump(dictionary,outfile)

    def _create_archive(self) -> None:
        self.archive_path = self.descriptor_path.joinpath("{}.zip".format(self.run_name))

        previous_path = Path.cwd()
        os.chdir(str(self.descriptor_local_path))

        with zipfile.ZipFile(str(self.archive_path), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for fpath in self.descriptor_local_path.rglob("*"):
                relative_path = fpath.relative_to(self.descriptor_local_path)
                zf.write(str(relative_path))

        zf.close()
        os.chdir(str(previous_path))

    def load_from_archive(self, archive_path: str = None) -> dict:
        archive_path = self._string_to_path(archive_path)

        archive = zipfile.ZipFile(str(archive_path))

        if archive.testzip() is not None:
            logging.warning(
                "archive may be corrupted, found invalid files {}".format(
                    archive.testzip()))

        yml_files = [fname for fname in archive.namelist() if "yml" in fname]

        assert len(yml_files) > 0, "Can't find .yml config in archive {}".format(archive_path)

        yml_file = archive.read(yml_files[0])
        return yaml.load(yml_file, Loader=yaml.FullLoader)

        # extract archive

        # populate config

    @staticmethod
    def _string_to_path(in_string: str) -> Path:
        return Path(in_string) if in_string is not None else None
