import pickle
from pathlib import Path
from pathlib import PosixPath

import numpy as np
from skimage import io as skio
# import skimage.external.tifffile as tifffile
import tifffile
import zetastitcher

SUPPORTED_IMG_FORMATS = ["tif", "tiff", "png", "jpg", "jpeg"]
SUPPORTED_STACK_FORMATS = ["tif", "tiff"]


def load_volume(imgpath,
                ignore_last_channel=False,
                data_mode="stack",
                channel_names=None,
                return_norm=False):
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
        return sorted(img_paths)

    def adjust_dimensions(vol):
        vol_shape = vol.shape
        # my target is [z, y, x, ch]
        if len(vol_shape) == 2:
            # this is a single monochromatic image
            # [y, x]
            # shoudln't happen that a dataset has only 1 image
            vol = np.expand_dims(vol, axis=0)  # adding z
            vol = np.expand_dims(vol, axis=-1)  # adding ch
        elif len(vol_shape) == 3:
            # 3d tensor without channels or 2D tensor with channel
            # [z, y, x] OR [y, x, ch]
            if data_mode == "single_images":
                z_len = len(img_paths)
                if z_len > 1:
                    # [z, y, x] case
                    vol = np.expand_dims(vol, axis=-1)  # adding ch
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
            vol = vol[..., :2]
        return vol

    def get_norm(vol):
        dtype = vol.dtype
        if dtype in [float, np.float, np.float16, np.float32, np.float64]:
            norm = np.finfo(dtype).max
        elif dtype in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            norm = np.iinfo(dtype).max
        return norm

    if data_mode == "stack":
        vol = skio.imread(str(imgpath), plugin="pil")
        if return_norm:
            norm = get_norm(vol)
            return postprocess_vol(vol), norm
        else:
            return postprocess_vol(vol)

    elif data_mode == "single_images":
        img_paths = glob_if_needed(imgpath)
        vol_list = []
        for slice_path in img_paths:
            img = skio.imread(slice_path, plugin="pil")
            vol_list.append(img)
        vol = np.array(vol_list)
        if return_norm:
            norm = get_norm(vol)
            return postprocess_vol(vol), norm
        else:
            return postprocess_vol(vol)

    elif data_mode == "zetastitcher":
        # channel_imgs = []

        # for chan_name in channel_names:
        #     channel_fpath = imgpath.joinpath(chan_name + ".zip")
        #     channel_imgs.append(zetastitcher.InputFile(channel_fpath)[...])

        # vol = np.stack(channel_imgs, axis=-1)
        # del channel_imgs

        vol = zetastitcher.InputFile(imgpath)[...]

        if return_norm:
            norm = get_norm(vol)
            return postprocess_vol(vol), norm
        else:
            return postprocess_vol(vol)

    elif data_mode == "multi_stack":
        img_paths = glob_if_needed(imgpath)
        vols_list = []
        for vol_path in img_paths:
            vol = skio.imread(vol_path, plugin="pil")
            vol = postprocess_vol(vol)
            vols_list.append(vol)

        if return_norm:
            norm = get_norm(vols_list[0])
            return vols_list, norm
        else:
            return vols_list
    else:
        raise ValueError(f"unsupported data_mode {data_mode}")


def save_volume(volume,
                output_path,
                fname="predictions",
                clip=True,
                save_tiff=True,
                save_8bit=True,
                save_pickle=True,
                append_tiff=False,
                return_outpaths=False):
    returns = []
    if save_pickle:
        pickle_out_path = output_path.joinpath(fname + ".pickle")

        with pickle_out_path.open(mode="wb") as out_file:
            pickle.dump(volume, out_file)
        returns.append(pickle_out_path)

    if clip:
        volume = np.clip(volume, a_min=0., a_max=1.)

    def exp_tiff(out_volume, name):
        if volume.shape[-1] == 2:
            pass
            # zeros = np.zeros_like(volume[...,0])
            # zeros = np.expand_dims(zeros, axis=-1)
            # volume = np.concatenate([volume, zeros], axis=-1)

        tiff_path = output_path.joinpath(name + ".tif")
        # skio.imsave(tiff_path, volume, plugin="pil", check_contrast=False)
        # tifffile.imsave(tiff_path, volume.astype(np.float32), photometric="minisblack")
        with tifffile.TiffWriter(str(tiff_path), append=append_tiff) as stack:
            for img_plane in out_volume:
                stack.save(img_plane)

        return tiff_path

    if save_tiff:
        tiff_path = exp_tiff(volume, name=fname)
        returns.append(tiff_path)
    if save_8bit:
        # vol_clipped = np.clip(volume, a_min=0., a_max=.99999)
        vol_8bit = (volume * 255).astype(np.uint8)
        tiff_8bit_path = exp_tiff(vol_8bit, name=fname + "_8bit")
        returns.append(tiff_8bit_path)

    return returns


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
