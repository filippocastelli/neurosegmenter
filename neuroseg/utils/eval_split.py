from argparse import ArgumentParser
from pathlib import Path
import csv

import numpy as np


def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        type=str,
                        action="store",
                        dest="dataset_path",
                        help="dataset path")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    ev = SplitEvaluator(dataset_path=dataset_path)


class SplitEvaluator:

    def __init__(self,
                 dataset_path: Path):
        self.dataset_path = dataset_path
        self.sections = ["train", "val", "test"]

        self.annotated_pxs = np.array([self.get_section_annotated_px(section) for section in self.sections])
        self.tot_px = np.sum(self.annotated_pxs)
        self.norm_annotated_px = self.annotated_pxs / self.tot_px

        print(f"train tot: {self.annotated_pxs[0]}, val tot: {self.annotated_pxs[1]}, test tot: {self.annotated_pxs[2]}")
        print(f"train %: {self.norm_annotated_px[0]}, val %: {self.norm_annotated_px[1]}, test %: {self.norm_annotated_px[2]}")

    def get_section_annotated_px(self, section: str) -> int:
        section_path = self.dataset_path.joinpath(section).joinpath("masks")
        csvs = section_path.glob("*.csv")

        annotated_px = 0
        for csv_path in csvs:
            bbox_list = self._parse_csv(csv_path)
            widht = bbox_list[2] - bbox_list[0]
            height = bbox_list[3] - bbox_list[1]
            annotated_px += widht*height

        return annotated_px

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
