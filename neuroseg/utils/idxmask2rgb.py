from argparse import ArgumentParser
from pathlib import Path
from shutil import copy

from skimage import io as skio
import numpy as np


def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        type=str,
                        action="store",
                        default="/home/phil/Scrivania/idxmask_dataset",
                        dest="dataset_path",
                        help="dataset path")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    i2rgb = IndexMask2RGB(dataset_path=dataset_path)


class IndexMask2RGB:
    def __init__(self,
                 dataset_path: Path,
                 class_values: tuple = (0, 1, 2)):

        self.dataset_path = dataset_path
        self.class_values = class_values

        self.sections = ["train", "val", "test"]
        for section in self.sections:
            self.convert_masks(section)

    def convert_masks(self, section: str) -> None:
        section_path = self.dataset_path.joinpath(section)
        section_mask_path = section_path.joinpath("masks")

        section_rgb_path = section_path.joinpath("masks_rgb")
        section_rgb_path.mkdir(exist_ok=True)

        tiff_paths = list(section_mask_path.glob("*.tif"))

        for fpath in tiff_paths:
            img = skio.imread(str(fpath), plugin="pil")

            img_stack = []
            for class_value in self.class_values:
                img_stack.append(np.where(img == class_value, 255, 0))
            out_img = np.stack(img_stack, axis=-1)

            out_class_img_path = section_rgb_path.joinpath(fpath.name)
            skio.imsave(str(out_class_img_path), out_img, plugin="pil")


if __name__ == "__main__":
    main()
