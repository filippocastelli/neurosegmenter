import numpy as np
from utils import load_volume, save_volume, is_supported_ext, glob_imgs
from config.config import SUPPORTED_STACK_FORMATS
from tensorflow.python.keras.models import load_model


class DataPredictorBase:
    def __init__(self, config, model=None):

        self.config = config
        self.ignore_last_channel = config.ignore_last_channel

        self.mode = self.config.config_type
        self._parse_settings()
        self._parse_paths()
        self._load_volume()
        self.prediction_model = self._load_model() if model is None else model
        self.predict()
        self._save_volume()

    def _parse_paths(self):
        if self.mode == "predict":
            if self.data_mode == "single_images":
                self.data_path = self.config.data_path
            elif self.data_mode == "stack":
                self.data_path = glob_imgs(
                    self.config.data_path, mode="stack", to_string=True
                )[0]
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
            self.chunk_size = self.config.chunk_size
            self.padding_mode = self.config.padding_mode
            self.n_output_classes = self.config.n_output_classes
            self.keep_tmp = self.config.keep_tmp
        elif self.mode == "training":
            # self.data_mode = self.config.ground_truth_mode
            self.data_mode = self.config.dataset_mode
            self.normalize_data = True
            self.output_mode = "stack"
            self.window_size = self.config.crop_shape
            self.batch_size = self.config.batch_size
            self.chunk_size = self.config.pe_chunk_size
            self.padding_mode = "reflect"
            self.keep_tmp = False
            self.n_output_classes = 1
        self.n_channels = self.config.n_channels

    def _load_model(self):
        return load_model(filepath=str(self.model_path), compile=False)

    # @staticmethod
    # def _glob_dir(dir_path):
    #     paths = [str(imgpath) for imgpath in sorted(dir_path.glob("*.*")) if ]

    @staticmethod
    def _load_single_volume(data_path, n_channels, data_mode, normalize_data=True):

        drop_last_channel = True if (n_channels == 2) else False
        expand_last_dim = True if n_channels == 1 else False

        vol = load_volume(
            data_path,
            drop_last_channel,
            expand_last_dim=expand_last_dim,
            data_mode=data_mode,
        )
        if normalize_data:
            max_norm = np.iinfo(vol.dtype).max
            vol = vol / max_norm

        return vol

    def _load_volume(self):
        if self.data_mode in ["single_images", "stack"]:

            self.input_data = self._load_single_volume(
                self.data_path, self.n_channels, self.data_mode
            )
            # drop_last_channel = True if (self.n_channels == 2) else False
            # expand_last_dim = True if self.n_channels == 1 else False

            # vol = load_volume(self.data_path,
            #                   drop_last_channel,
            #                   expand_last_dim=expand_last_dim,
            #                   data_mode=self.data_mode)

            # if self.normalize_data:
            #     max_norm = np.iinfo(vol.dtype).max
            #     vol = vol / max_norm

            # self.input_data = vol
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
        
        if self.data_mode in ["single_images", "stack"]:
            if self.output_mode == "stack":
                save_volume(
                    self.predicted_data, self.output_path, save_tiff=True, save_pickle=True
                )
            else:
                raise NotImplementedError(self.output_mode)
                
        elif self.data_mode == "multi_stack":
            # self.predicted data is a list of numpy arrays
            # self.data_paths is assigned in _load_volume
            # should be aligned with self.predicted_data
            for idx, input_data_fpath in enumerate(self.data_paths):
                
                fname = input_data_fpath.name
                
                save_volume(self.predicted_data[fname],
                            self.output_path,
                            fname=fname,
                            save_tiff=True,
                            save_pickle=True                            
                    )
            
            
            
            
