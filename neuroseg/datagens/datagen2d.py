import tensorflow as tf
from tensorflow.data import Dataset
from skimage import io as skio
import numpy as np

SUPPORTED_FORMATS = ["png", "tif", "tiff"]

class dataGen2D:
    def __init__(self,
                 config,
                 partition="train",
                 normalize=True,
                 verbose=False):
        
        self.config = config
        self.partition = partition
        self.data_path_dict = self.config.path_dict[partition]
        self.normalize = normalize
        
        self.verbose = verbose
        self._path_sanity_check()
        
        self.crop_shape = config.crop_shape
        self.batch_size = config.batch_size
        
        self.buffer_size = config.da_buffer_size
        self.debug_mode = config.da_debug_mode
        self.single_thread = config.da_single_thread
        self.threads = 1 if config.da_single_thread == True else config.da_threads
        
        self.transforms = config.da_transforms
        self.transform_cfg = config.da_transform_cfg
        
        # init sequence
        self._scan_dirs()
        self.data = self.gen_dataset()
        
        
    def _scan_dirs(self):
        self.frames_paths = self._glob_subdirs("frames")
        self.masks_paths = self._glob_subdirs("masks")
        
    def _glob_subdirs(self, subdir):
        subdir_paths = [str(imgpath) for imgpath in
                        sorted(self.data_path_dict[subdir].glob("*.*"))
                        if self._is_supported_format(imgpath)]
        
        if self.verbose:
            print("there are {} {} imgs".format(len(subdir_paths), subdir))
            
        return subdir_paths
    
    @staticmethod
    def _is_supported_format(fpath):
        extension = fpath.suffix.split(".")[1]
        return extension in SUPPORTED_FORMATS
        
    def _path_sanity_check(self):
        if not (self.data_path_dict["frames"].is_dir()
                and self.data_path_dict["masks"].is_dir()):
            raise ValueError("dataset paths are not actual dirs")
            
    def _get_img_shape(self):
        frame_path = self.frames_paths[0]
        mask_path = self.masks_paths[0]
        frame = self._load_img(frame_path)
        mask = self._load_img(mask_path)
        return frame.shape, mask.shape
            
    def gen_dataset(self):
        
        ds = Dataset.from_tensor_slices((self.frames_paths, self.masks_paths))
        ds = ds.map(self._load_example_wrapper,
                    deterministic=True,
                    num_parallel_calls=self.threads)
        frame_shape, mask_shape = self._get_img_shape()
        ds = ds.map(map_func=lambda frame, mask: self._set_shapes(frame, mask, frame_shape, mask_shape))
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.buffer_size)
        return ds
    
    @classmethod
    def _load_example(cls, frame_path, mask_path):
        frame = cls._load_img(frame_path.numpy().decode("utf-8"))
        mask = cls._load_img(mask_path.numpy().decode("utf-8"))
        return frame, mask
    
    @staticmethod
    def _load_img(img_to_load_path, normalize=True):
        try:
            img = skio.imread(img_to_load_path)
            if normalize:
                norm_constant = np.iinfo(img.dtype).max
                img = img/norm_constant
                return img
        except ValueError:
            raise ValueError("This image failed: {}, check for anomalies".format(str(img_to_load_path)))
    
    @classmethod
    def _load_example_wrapper(cls, frame_path, mask_path):
        file = tf.py_function(cls._load_example, [frame_path, mask_path], (tf.float64, tf.float64))
        return file
    
    @staticmethod
    def _set_shapes(frame, mask, frame_shape, mask_shape):
        frame.set_shape(frame_shape)
        mask.set_shape(mask_shape)
        return frame, mask