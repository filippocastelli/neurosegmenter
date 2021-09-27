from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import numpy as np
import tifffile

from skimage import io as skio
from neuroseg.utils.idxmask2rgb import IndexMask2RGB


def main():
    parser = ArgumentParser()

    parser.add_argument("-i", "--img",
                        type=str,
                        action="store",
                        dest="img_path",
                        help="image path")

    args = parser.parse_args()
    in_img_path = Path(args.img_path)

    img = tifffile.imread(str(in_img_path))
    rgb_img = toargmax(img)
    out_fpath = in_img_path.parent.joinpath(in_img_path.name + ".rgb.tif")
    skio.imsave(str(out_fpath), rgb_img, plugin="pil")


def toargmax(img: np.ndarray = None,
             class_values: tuple = (0, 1, 2),
             pos_value: Union[int, float] = 1):
    img = np.argmax(img, axis=-1)  # z, y, x, values are 0, 1, 2, 3
    rgb_img = IndexMask2RGB.to_rgb(img, class_values, pos_value=pos_value)

    return rgb_img



if __name__ == "__main__":
    main()
