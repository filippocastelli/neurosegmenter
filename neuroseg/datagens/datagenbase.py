from typing import Union
from pathlib import Path

from neuroseg.utils import SUPPORTED_IMG_FORMATS, SUPPORTED_STACK_FORMATS
from neuroseg.config import TrainConfig, PredictConfig


class DataGenBase:
    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 partition: str = "train",
                 data_augmentation: bool = True,
                 verbose: bool = False):

        self.config = config
        self.partition = partition
        self.data_path_dict = self.config.path_dict[partition]
        self.dataset_mode = self.config.dataset_mode
        self.use_bboxes = self.config.use_bboxes

        # TODO: deprecate positive_class_value and negative_class_value
        self.positive_class_value = self.config.positive_class_value
        self.verbose = verbose
        self._path_sanity_check()

        self.crop_shape = config.window_size
        self.batch_size = config.batch_size

        self.normalize_inputs = self.config.normalize_inputs
        self.normalize_masks = self.config.normalize_masks
        self.soft_labels = self.config.soft_labels

        self.single_thread = config.da_single_thread
        self.threads = 1 if config.da_single_thread == True else config.da_threads

        self.data_augmentation = data_augmentation
        self.transforms = config.da_transforms
        self.transform_cfg = config.da_transform_cfg

        self.class_values = config.class_values
        self.background_value = config.background_value
        # init sequence
        self._scan_dirs()

    def _path_sanity_check(self) -> None:
        if self.dataset_mode in ["single_images", "stack"]:
            if not (self.data_path_dict["frames"].is_dir()
                    and self.data_path_dict["masks"].is_dir()):
                raise ValueError("dataset paths are not actual dirs")

    def _scan_dirs(self) -> None:
        if self.dataset_mode in ["single_images", "stack", "multi_stack"]:
            self.frames_paths = self._glob_subdirs("frames")
            self.masks_paths = self._glob_subdirs("masks")
        else:
            pass

    def _glob_subdirs(self, subdir: str) -> list:
        subdir_paths = [str(imgpath) for imgpath in
                        sorted(self.data_path_dict[subdir].glob("*.*"))
                        if self._is_supported_format(imgpath)]
        if self.verbose:
            print("there are {} {} imgs".format(len(subdir_paths), subdir))
        return subdir_paths

    @staticmethod
    def _is_supported_format(fpath: Path) -> bool:
        extension = fpath.suffix.split(".")[1]
        return extension in SUPPORTED_IMG_FORMATS or extension in SUPPORTED_STACK_FORMATS
