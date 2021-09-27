from argparse import ArgumentParser
from pathlib import Path
import csv

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
    cb = ClassBalanceEval(dataset_path=dataset_path)


class ClassBalanceEval:
    def __init__(self,
                 dataset_path: Path,
                 class_values: tuple = (0, 1, 2, 255),
                 ):
        self.dataset_path = dataset_path
        self.dataset_sections = ["train", "val", "test"]

        self.class_values = class_values
        self.px_count = self.get_img_data()

        for class_value in self.class_values:
            class_count = self.px_count[class_value]
            norm_count = class_count / self.px_count["tot"]

            print(f"{class_value=}, count={class_count}, normalized_count={norm_count}, weight={np.log(1/norm_count)}")

        print(self.px_count)
#        self.background_value = background_value

    def get_img_data(self):
        px_dict = {"tot": 0}
        for class_value in self.class_values:
            px_dict[class_value] = 0

        for section in self.dataset_sections:
            section_path = self.dataset_path.joinpath(section)
            masks_dir = section_path.joinpath("masks")

            masks_files = list(masks_dir.glob("*.tif"))
            csv_files = [fpath.parent.joinpath(fpath.name+".csv") for fpath in masks_files]

            for idx, fpath in enumerate(masks_files):
                img = skio.imread(str(fpath), plugin="pil")
                bbox = self._parse_csv(csv_files[idx])

                cropped_img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                uniques, counts = np.unique(cropped_img, return_counts=True)
                px_dict_img = dict(zip(uniques, counts))

                tot_px = np.prod(cropped_img.shape)
                px_dict["tot"] += tot_px

                for class_value in self.class_values:
                    if class_value not in px_dict_img:
                        pass
                    else:
                        px_dict[class_value] += px_dict_img[class_value]

        return px_dict



    @staticmethod
    def _parse_csv(csv_path: Path) -> list:
        """return the first row of a csv file"""
        out_list = []
        with csv_path.open(mode="r") as infile:
            reader = csv.reader(infile)
            for row in reader:
                row_ints = [int(elem) for elem in row]
                out_list.append(row_ints)
        return out_list[0]


if __name__ == "__main__":
    main()