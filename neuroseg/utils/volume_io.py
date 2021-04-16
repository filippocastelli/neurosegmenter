import pickle
from pathlib import Path
from pathlib import PosixPath

import numpy as np
from skimage import io as skio
# import skimage.external.tifffile as tifffile
import tifffile

SUPPORTED_IMG_FORMATS = ["tif", "tiff", "png", "jpg", "jpeg"]
SUPPORTED_STACK_FORMATS = ["tif", "tiff"]

def load_volume(imgpath,
                ignore_last_channel=False,
                data_mode="stack"):
    
    # if not is_supported_ext(imgpath):
    #     raise ValueError(imgpath, "unsupported image format")
        
    def glob_if_needed(imgpath):
        if type(imgpath) is list:
            img_paths = imgpath
        elif type(imgpath) is PosixPath:
            if imgpath.is_dir():
                img_paths = [fpath for fpath in imgpath.glob("*.*") if is_supported_ext(fpath)]
            else:
                img_paths = [imgpath]
        else:
            raise TypeError("invalid type for imgpath")
        return img_paths
    
    def adjust_dimensions(vol):
        vol_shape = vol.shape
        # my target is [z, y, x, ch]
        if len(vol_shape) == 2:
            # this is a single monochromatic image
            # [y, x]
            # shoudln't happen that a dataset has only 1 image
            vol = np.expand_dims(vol, axis=0) # adding z
            vol = np.expand_dims(vol, axis=-1) # adding ch
        elif len(vol_shape) == 3:
            # 3d tensor without channels or 2D tensor with channel
            # [z, y, x] OR [y, x, ch]
            if data_mode == "single_images":
                z_len = len(img_paths)
                if z_len > 1:
                    # [z, y, x] case
                    vol = np.expand_dims(vol, axis=-1) # adding ch
                else:
                    # [y, x, ch] case
                    vol = np.expand_dims(vol, axis=0)
            elif data_mode in ["stack", "multi_stack"]:
                z_len = vol_shape[0]
                if z_len > 1:
                    # [z, y, x] case
                    vol = np.expand_dims(vol, axis=-1)
                else:
                    # [y, x, ch] case
                    vol = np.expand_dims(vol, axis=0)
        elif len(vol_shape) == 4:
            # 3d tensor with channel
            # should already be [z, y, x, ch]
            # you're beautiful as you are, my little tensor
            pass
        else:
            # boi, you're an abomination in the eyes of men and God
            # this training has to fail because of you
            # cursed you are among your fellow tensors
            raise ValueError("input volume has too many dims")
        return vol
        
    def postprocess_vol(vol):
        vol = adjust_dimensions(vol)
        if ignore_last_channel:
            vol_shape = vol.shape
            assert len(vol_shape) >= 3, "this image does not have a channel dim"
            vol = vol[...,:2]
        return vol
    
    if data_mode == "stack":
        vol = skio.imread(str(imgpath), plugin="pil")
        return postprocess_vol(vol)
    elif data_mode == "single_images":
        img_paths = glob_if_needed(imgpath)
        vol_list = []
        for slice_path in img_paths:
            img = skio.imread(slice_path, plugin="pil")
            vol_list.append(img)
        vol = np.array(vol_list)
        return postprocess_vol(vol)
    elif data_mode == "multi_stack":
        img_paths = glob_if_needed(imgpath)
        vols_list = []
        for vol_path in img_paths:
            vol = skio.imread(vol_path, plugin="pil")
            vol = postprocess_vol(vol)
            vols_list.append(vol)
        return vols_list
    else:
        raise ValueError(f"unsupported data_mode {data_mode}")

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
    