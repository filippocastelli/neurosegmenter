from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tifffile

from skimage import io as skio
from idxmask2rgb import IndexMask2RGB


def main():
    parser = ArgumentParser()

    parser.add_argument("-i", "--img",
                        type=str,
                        action="store",
                        dest="img_path",
                        help="image path")

    args = parser.parse_args()
    in_img_path = Path(args.img_path)
    toargmax(in_img_path=in_img_path)


def toargmax(in_img_path: Path,
             class_values: tuple = (0, 1, 2)):
    img = tifffile.imread(str(in_img_path))
    img = np.argmax(img, axis=-1)  # z, y, x, values are 0, 1, 2, 3

    rgb_img = IndexMask2RGB.to_rgb(img, class_values)
    out_fpath = in_img_path.parent.joinpath(in_img_path.name + ".rgb.tif")
    skio.imsave(str(out_fpath), rgb_img, plugin="pil")


if __name__ == "__main__":
    main()
