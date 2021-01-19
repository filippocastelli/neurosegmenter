import pickle
from pathlib import Path

import numpy as np
from skimage import io as skio
# import skimage.external.tifffile as tifffile
import tifffile

SUPPORTED_IMG_FORMATS = ["tif", "tiff", "png", "jpg", "jpeg"]
SUPPORTED_STACK_FORMATS = ["tif", "tiff"]

def load_volume(imgpath,
                drop_last_dim=True,
                expand_last_dim=False,
                squeeze=False,
                data_mode="stack"):
    
    # if not is_supported_ext(imgpath):
    #     raise ValueError(imgpath, "unsupported image format")
        
    if data_mode == "stack":
        vol = skio.imread(imgpath, plugin="pil")
    elif data_mode == "single_images":
        img_paths = [fpath for fpath in imgpath.glob("*.*") if is_supported_ext(fpath)]
        vol_list = []
        for slice_path in img_paths:
            img = skio.imread(slice_path, plugin="pil")
            vol_list.append(img)
        
        vol = np.array(vol_list)
    else:
        raise ValueError("unsupported data_mode: {}".format(data_mode))
    if drop_last_dim:
        vol = vol[...,:2]
    if expand_last_dim:
        vol = np.expand_dims(vol, axis=-1)
        
    if squeeze:
        vol = np.squeeze(vol)
        
    return vol

def save_volume(volume,
                output_path,
                fname="predictions",
                save_tiff=True,
                save_pickle=True):
    
    if save_pickle:
        pickle_out_path = output_path.joinpath(fname + ".pickle")

        with pickle_out_path.open(mode="wb") as out_file:
            pickle.dump(volume, out_file)
    
    if save_tiff:
        
        if volume.shape[-1] == 2:
            pass
            # zeros = np.zeros_like(volume[...,0])
            # zeros = np.expand_dims(zeros, axis=-1)
            # volume = np.concatenate([volume, zeros], axis=-1)
        
        tiff_path = output_path.joinpath(fname + ".tif")
        # skio.imsave(tiff_path, volume, plugin="pil", check_contrast=False)
        # tifffile.imsave(tiff_path, volume.astype(np.float32), photometric="minisblack")
        with tifffile.TiffWriter(str(tiff_path)) as stack:
            for img_plane in volume:
                stack.save(img_plane)
        
def is_supported_ext(path, mode="img"):
    suffix = path.suffix
    extension = suffix.split(".")[1]
    if mode == "img":
        return extension in SUPPORTED_IMG_FORMATS
    elif mode == "stack":
        return extension in SUPPORTED_STACK_FORMATS
    else:
        raise ValueError("mode {} is not a valid input mode".format(mode))
        
        
def glob_imgs(dir_path, mode="stack", to_string=False):
    dir_path = Path(dir_path)
    paths = [imgpath for imgpath in sorted(dir_path.glob("*.*")) if is_supported_ext(imgpath, mode=mode)]
    
    if to_string:
        paths = [str(path) for path in paths]
        
    return paths
    