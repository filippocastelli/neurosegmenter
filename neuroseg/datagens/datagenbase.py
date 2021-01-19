from utils import SUPPORTED_IMG_FORMATS, SUPPORTED_STACK_FORMATS

class dataGenBase:
    def __init__(self,
                 config,
                 partition="train",
                 data_augmentation=True,
                 normalize_inputs=True,
                 verbose=False):
        
        self.config = config
        self.partition = partition
        self.data_path_dict = self.config.path_dict[partition]
        self.dataset_mode = self.config.dataset_mode
        
        self.positive_class_value = self.config.positive_class_value
        self.verbose = verbose
        self._path_sanity_check()
        
        self.crop_shape = config.crop_shape
        self.batch_size = config.batch_size
        
        self.normalize_inputs = normalize_inputs
        
        self.single_thread = config.da_single_thread
        self.threads = 1 if config.da_single_thread == True else config.da_threads
        
        self.data_augmentation = data_augmentation
        self.transforms = config.da_transforms
        self.transform_cfg = config.da_transform_cfg
        
        # init sequence
        self._scan_dirs()
        
    
    def _path_sanity_check(self):
        
        if self.dataset_mode in ["single-images", "stack"]:
            if not (self.data_path_dict["frames"].is_dir()
                    and self.data_path_dict["masks"].is_dir()):
                raise ValueError("dataset paths are not actual dirs")
            
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
        return extension in SUPPORTED_IMG_FORMATS or extension in SUPPORTED_STACK_FORMATS