import pickle

import numpy as np
from skimage import io as skio
# import skimage.external.tifffile as tifffile
import tifffile

SUPPORTED_FORMATS = ["tif"]

def load_volume(imgpath,
                drop_last_dim=True,
                expand_last_dim=False):
    
    if not is_supported_ext(imgpath):
        raise ValueError(imgpath, "unsupported image format")
    
    vol = skio.imread(imgpath, plugin="pil")
    if drop_last_dim:
        vol = vol[...,:2]
    if expand_last_dim:
        vol = np.expand_dims(vol, axis=-1)
        
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
        
def is_supported_ext(path):
    suffix = path.suffix
    extension = suffix.split(".")[1]
    return extension in SUPPORTED_FORMATS
    