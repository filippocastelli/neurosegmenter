from argparse import ArgumentParser
from shutil import copy
from pathlib import Path
from sklearn.model_selection import train_test_split

from tqdm import tqdm


def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        type=str,
                        action="store",
                        dest="dataset_path",
                        help="dataset path")

    parser.add_argument("--train",
                        type=float,
                        action="store",
                        default=0.7,
                        dest="train_ratio",
                        help="train ratio")

    parser.add_argument("--test",
                        type=float,
                        action="store",
                        default=0.2,
                        dest="test_ratio",
                        help="test ratio")

    parser.add_argument("--val",
                        type=float,
                        action="store",
                        default=0.1,
                        dest="val_ratio",
                        help="val ratio")

    parser.add_argument("-m", "--mode",
                        type=str,
                        action="store",
                        default="normal",
                        dest="dataset_mode",
                        help="dataset mode selection [\"normal\", \"csv\"]")

    parser.add_argument("--frame_pattern",
                        type=str,
                        action="store",
                        default="_frame",
                        dest="frame_pattern",
                        help="frame pattern")

    parser.add_argument("--mask_pattern",
                        type=str,
                        action="store",
                        default="_mask",
                        dest="mask_pattern",
                        help="mask pattern")
    
    parser.add_argument("-e", "--extension",
                        type=str,
                        action="store",
                        default="tif",
                        dest="extension",
                        help="extension")

    parser.add_argument("-n", "--no_pattern_out",
                        action="store_true",
                        default=False,
                        dest="no_pattern_out",
                        help="substitute pattern with empty str in output fnames")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    train_ratio = float(args.train_ratio)
    test_ratio = float(args.test_ratio)
    val_ratio = float(args.val_ratio)

    dataset_mode = args.dataset_mode
    frame_pattern = args.frame_pattern
    mask_pattern = args.mask_pattern
    extension = args.extension
    no_pattern_out = args.no_pattern_out

    if dataset_mode == "normal":
        ds = DatasetSplitter(
            dataset_path=dataset_path,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            frame_pattern=frame_pattern,
            mask_pattern=mask_pattern,
            extension=extension,
            no_pattern_out=no_pattern_out,
        )
    elif dataset_mode == "csv":
        ds = CSVDatasetSplitter(
            dataset_path=dataset_path,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio
        )
    else:
        raise ValueError("dataset mode not supported")


class CSVDatasetSplitter:
    def __init__(self,
                 dataset_path: Path,
                 train_ratio: float = 0.7,
                 test_ratio: float = 0.2,
                 val_ratio: float = 0.1,
                 ):
        self.dataset_path = dataset_path

        ratios_sum = train_ratio + test_ratio + val_ratio
        self.train_ratio = train_ratio / ratios_sum
        self.test_ratio = test_ratio / ratios_sum
        self.val_ratio = val_ratio / ratios_sum

        self.img_list = sorted(list(self.dataset_path.joinpath("frames").glob("*.tif")))
        self.mask_list = sorted(list(self.dataset_path.joinpath("masks").glob("*.tif")))
        self.csv_list = [fpath.parent.joinpath(fpath.name + ".csv") for fpath in self.mask_list]

        self.split_dict = self.split_dataset()
        self.copy_ds(self.split_dict)

    def split_dataset(self) -> dict:
        train_img, test_img, train_mask, test_mask, train_csv, test_csv = train_test_split(self.img_list,
                                                                                           self.mask_list,
                                                                                           self.csv_list,
                                                                                           train_size=self.train_ratio)

        norm = self.test_ratio + self.val_ratio
        split_ratio = self.test_ratio/norm

        test_img, val_img, test_mask, val_mask, test_csv, val_csv = train_test_split(test_img, test_mask, test_csv,
                                                                                     train_size=split_ratio)

        return {
            "train": (train_img, train_mask, train_csv),
            "test": (test_img, test_mask, test_csv),
            "val": (val_img, val_mask, val_csv)
        }

    def copy_ds(self, split_dict: dict):
        for section, arg_tuple in split_dict.items():
            section_dir = self.dataset_path.joinpath(section)
            section_dir.mkdir(exist_ok=True)

            frames_subdir = section_dir.joinpath("frames")
            frames_subdir.mkdir(exist_ok=True)
            masks_subdir = section_dir.joinpath("masks")
            masks_subdir.mkdir(exist_ok=True)

            for idx, img_fpath in enumerate(arg_tuple[0]):
                mask_fpath = arg_tuple[1][idx]
                csv_fpath = arg_tuple[2][idx]

                out_img = frames_subdir.joinpath(img_fpath.name)
                out_mask = masks_subdir.joinpath(mask_fpath.name)
                out_csv = masks_subdir.joinpath(csv_fpath.name)

                copy(str(img_fpath), str(out_img))
                copy(str(mask_fpath), str(out_mask))
                copy(str(csv_fpath), str(out_csv))

class DatasetSplitter:
    def __init__(self,
                 dataset_path: Path,
                 train_ratio: float = 0.7,
                 test_ratio: float = 0.2,
                 val_ratio: float = 0.1,
                 extension: str = "tiff",
                 frame_pattern: str = "_frame",
                 mask_pattern: str = "_annotation",
                 no_pattern_out: bool = False
                 ):

        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.frame_pattern = frame_pattern
        self.mask_pattern = mask_pattern

        self.no_pattern_out = no_pattern_out

        self.tiff_paths = list(self.dataset_path.glob(f"*.{extension}"))

        self.img_paths = [fpath for fpath in self.tiff_paths if self.frame_pattern in fpath.name]
        self.mask_paths = [fpath for fpath in self.tiff_paths if self.mask_pattern in fpath.name]

        self.split_dict = self.split_dataset()
        self.copy_ds(self.split_dict)

    def split_dataset(self):
        
        train_img, test_img, train_mask, test_mask  = train_test_split(
            self.img_paths,
            self.mask_paths,
            train_size=self.train_ratio
            )
        norm = self.test_ratio + self.val_ratio
        split_ratio = self.test_ratio/norm

        test_img, val_img, test_mask, val_mask = train_test_split(
            test_img,
            test_mask,
            train_size=split_ratio
        )

        return {
            "train": (train_img, train_mask),
            "test": (test_img, test_mask),
            "val": (val_img, val_mask)
        }
    
    def copy_ds(self, split_dict: dict):
        for section, arg_tuple in split_dict.items():
            print("copying", section)
            section_dir = self.dataset_path.joinpath(section)
            section_dir.mkdir(exist_ok=True)

            frames_subdir = section_dir.joinpath("frames")
            frames_subdir.mkdir(exist_ok=True)
            masks_subdir = section_dir.joinpath("masks")
            masks_subdir.mkdir(exist_ok=True)

            for idx, img_fpath in enumerate(tqdm(arg_tuple[0])):
                mask_fpath = arg_tuple[1][idx]
                
                img_name = img_fpath.name
                mask_name = mask_fpath.name

                if self.no_pattern_out:
                    img_name = img_name.replace(self.frame_pattern, "")
                    mask_name = mask_name.replace(self.mask_pattern, "")

                out_img = frames_subdir.joinpath(img_name)
                out_mask = masks_subdir.joinpath(mask_name)

                print("Copying", img_fpath, "to", out_img)
                copy(str(img_fpath), str(out_img))
                copy(str(mask_fpath), str(out_mask))

if __name__ == "__main__":
    main()
