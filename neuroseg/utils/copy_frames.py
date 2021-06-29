from argparse import ArgumentParser
from pathlib import Path
from shutil import copy


def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        type=str,
                        action="store",
                        dest="dataset_path",
                        help="dataset path")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    cf = CopyFrames(dataset_path=dataset_path)


class CopyFrames:

    def __init__(self,
                 dataset_path: Path):
        self.dataset_path = dataset_path
        self.sections = ["train", "val", "test"]
        for section in self.sections:
            self.copy_frames(section)

    def copy_frames(self, section: str) -> None:
        section_path = self.dataset_path.joinpath(section)
        section_mask_path = section_path.joinpath("masks")
        section_frames_path = section_path.joinpath("frames")
        section_frames_path.mkdir(exist_ok=True)

        tiff_paths = section_mask_path.glob("*.tif")

        common_frame_dir = self.dataset_path.joinpath("frames")
        frame_source_fpaths = [common_frame_dir.joinpath(fpath.name) for fpath in tiff_paths]

        for frame_src in frame_source_fpaths:
            frame_dest = section_frames_path.joinpath(frame_src.name)
            copy(str(frame_src), str(frame_dest))


if __name__ == "__main__":
    main()
