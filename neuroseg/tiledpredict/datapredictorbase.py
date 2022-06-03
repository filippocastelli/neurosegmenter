import numpy as np
from typing import Union
import h5py

from tensorflow.python.keras.models import load_model

from neuroseg.utils import load_volume, save_volume, glob_imgs
# from neuroseg.config.config import SUPPORTED_STACK_FORMATS
from neuroseg.config import TrainConfig, PredictConfig

class DataPredictorBase:
    def __init__(self, config: Union[PredictConfig, TrainConfig], model=None, in_fpath=None):

        self.config = config
        self.mode = self.config.config_type
        self.to_segmentation = self.config.to_segmentation
        self.in_fpath = in_fpath

        if self.mode == "predict":
            if self.config.multi_gpu or self.config.pe_multigpu:
                self.multi_gpu = True
        else:
            self.multi_gpu = False

        self.n_tiling_threads = self.config.n_tiling_threads
        self._parse_settings()
        self._parse_paths()
        self._load_volume()
        self.prediction_model = self._load_model() if model is None else model
        self.predict()
        self._save_volume()

    def _parse_paths(self):
        if self.mode == "predict":
            self.channel_names = None
            if self.data_mode == "single_images":
                self.data_path = self.config.data_path
            elif self.data_mode == "stack":
                if self.config.data_path.is_file():
                    self.data_path = self.config.data_path
                elif self.config.data_path.is_dir():
                    self.data_path = glob_imgs(
                        self.config.data_path, mode="stack", to_string=True
                    )[0]
                else:
                    raise ValueError(f"invalid data path {str(self.data_path)}")

            elif self.data_mode == "zetastitcher":
                self.data_path = self.config.data_path
                self.channel_names = self.config.channel_names
                # self.in_fpath overrides config file setting

                if self.in_fpath is not None:
                    self.data_path = self.in_fpath

            elif self.data_mode == "multi_stack":
                self.data_path = self.config.data_path
                # raise NotImplementedError(self.data_mode)
            else:
                raise NotImplementedError(self.data_mode)
            self.model_path = self.config.model_path

        elif self.mode == "training":
            if self.data_mode == "single_images":
                self.data_path = self.config.test_paths["frames"]
            elif self.data_mode == "stack":
                # data_path must point to a file
                self.data_path = glob_imgs(
                    self.config.test_paths["frames"], mode="stack", to_string=True
                )[0]
            elif self.data_mode == "h5_dataset":
                self.data_path = self.config.path_dict["test"]
            elif self.data_mode == "multi_stack":
                self.data_path = self.config.test_paths["frames"]
                # raise NotImplementedError(self.data_mode)
            else:
                raise NotImplementedError(self.data_mode)

        self.temp_path = self.config.temp_path
        self.output_path = self.config.output_path

    def _parse_settings(self):
        if self.mode == "predict":
            self.data_mode = self.config.data_mode
            self.normalize_data = self.config.normalize_data
            self.window_size = self.config.window_size
            self.output_mode = self.config.output_mode
            self.batch_size = self.config.batch_size
            # self.chunk_size = self.config.chunk_size
            self.padding_mode = self.config.padding_mode
            self.n_output_classes = self.config.n_output_classes
            # self.keep_tmp = self.config.keep_tmp
        elif self.mode == "training":
            # self.data_mode = self.config.ground_truth_mode
            self.data_mode = self.config.dataset_mode
            self.normalize_data = self.config.normalize_inputs
            self.output_mode = self.config.output_mode
            self.window_size = self.config.window_size
            self.batch_size = self.config.batch_size
            # self.chunk_size = self.config.pe_chunk_size
            self.padding_mode = "reflect"
            # self.keep_tmp = False
            self.n_output_classes = self.config.n_output_classes
        self.n_channels = self.config.n_channels
        self.extra_padding_windows = self.config.extra_padding_windows
        self.tiling_mode = self.config.tiling_mode
        self.window_overlap = self.config.window_overlap
        self.debug = self.config.predict_inspector

    def _load_model(self):
        return load_model(filepath=str(self.model_path), compile=False)

    # @staticmethod
    # def _glob_dir(dir_path):
    #     paths = [str(imgpath) for imgpath in sorted(dir_path.glob("*.*")) if ]

    @staticmethod
    def _load_single_volume(data_path, n_channels, data_mode, normalize_data=True, channel_names=None):
        drop_last_channel = True if (n_channels == 2) else False

        vol = load_volume(
            data_path,
            ignore_last_channel=drop_last_channel,
            data_mode=data_mode,
            channel_names=channel_names
        )
        if normalize_data:
            max_norm = np.iinfo(vol.dtype).max
            vol = vol / max_norm

        return vol

    def _load_volume(self):
        if self.data_mode in ["single_images", "stack", "zetastitcher"]:
            self.input_data = self._load_single_volume(
                data_path=self.data_path, n_channels=self.n_channels, data_mode=self.data_mode, channel_names=self.channel_names
            )
        elif self.data_mode == "h5_dataset":
            h5file = h5py.File(str(self.data_path), "r")
            # n_vols = len(h5file["data"])
            # loaded_vols = []
            # for vol_idx in range(n_vols):
            #     vol = h5file["data"][vol_idx, 0, ...]
            #     if self.normalize_data:
            #         norm = np.iinfo(vol.dtype).max
            #         vol = vol / norm
            #     loaded_vols.append(vol)
            self.input_data = h5file["data"]

        elif self.data_mode == "multi_stack":
            self.data_paths = glob_imgs(self.data_path, mode="stack")
            loaded_vols = []
            for data_fpath in self.data_paths:
                loaded_vols.append(
                    self._load_single_volume(data_fpath, self.n_channels, "stack")
                )

            self.input_data = loaded_vols
        else:
            raise NotImplementedError(self.data_mode)

    def _save_volume(self):
        if self.output_mode == "single_images":
            raise NotImplementedError(self.output_mode)
        elif self.output_mode == "stack":
            save_volume(
                self.predicted_data, self.output_path, save_tiff=True, save_pickle=True
            )     
        elif self.output_mode == "multi_stack":
            # self.predicted data is a list of numpy arrays
            # self.data_paths is assigned in _load_volume
            # should be aligned with self.predicted_data
            for idx, input_data_fpath in enumerate(self.data_paths):
                
                fname = input_data_fpath.stem
                key = input_data_fpath.name
                
                save_volume(self.predicted_data[key],
                            self.output_path,
                            fname=fname,
                            save_tiff=True,
                            save_pickle=True                            
                    )

        # this occupies way too much resources
        # best solution would be to override save_volume() in H5DataPredictor
        # and save each volume in the main predict loop
        elif self.output_mode == "h5_dataset":
            for idx, vol in enumerate(self.predicted_data):
                fname = str(f"{idx}_predict")
                save_volume(volume=vol,
                            output_path=self.output_path,
                            fname=fname,
                            save_tiff=True,
                            save_pickle=True)
        else:
            raise NotImplementedError(self.data_mode)

    @staticmethod
    def _get_autocrop_range(vol: np.ndarray, grad_threshold:float=0.01):
        """
        Returns the range of indices to horizontally crop the volume.
        :param vol: 3D volume
        :param grad_threshold: threshold for the gradient of the volume
        :return: (start_idx, end_idx)
        """
        # vol is assumed to be [z, y, x, ch]
        if len(vol.shape) == 4:
            vol = np.sum(vol, axis=-1)
            assert len(vol.shape) == 3, "vol.shape = {}".format(vol.shape)
        profile = np.sum(vol, axis=0)
        profile = np.sum(profile, axis=0)
        gradient = np.gradient(profile)

        gradient = gradient / gradient.max()

        filtered_grad = np.where(np.abs(gradient) > grad_threshold, gradient, 0)
        start_idx = np.min(np.nonzero(filtered_grad))
        end_idx = len(filtered_grad) - np.min(np.nonzero(np.flip(filtered_grad)))
        
        return start_idx, end_idx