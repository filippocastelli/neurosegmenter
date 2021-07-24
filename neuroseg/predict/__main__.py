from argparse import ArgumentParser
from pathlib import Path
import logging

from skimage import io as skio
import numpy as np
import tifffile

from neuroseg import PredictConfig, predict
from tqdm import tqdm


def setup_logger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)


def main():
    parser = ArgumentParser()

    parser.add_argument("-c", "--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/examples/config_predict.yml",
                        help="Configuration file path")

    parser.add_argument("-f", "--fpath", action="store", type=str,
                        dest="input_fpath",
                        default="/home/phil/Scrivania/substack2",
                        help="Input file path")

    parser.add_argument("-o", "--out", action="store", type=str,
                        dest="output_path",
                        default="/home/phil/Scrivania/out_pred",
                        help="Output path")
    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.configuration_path_str)
    in_fpath = Path(args.input_fpath) if args.input_fpath is not None else None
    out_fpath = Path(args.output_path) if args.output_path is not None else None

    config = PredictConfig(cfg_path)
    class_values = config.class_values
    predicted_data = predict(config, in_fpath=in_fpath)

    for img_plane, img in enumerate(tqdm(predicted_data)):

        for idx, class_value in enumerate(class_values):
            out_fpath_class = out_fpath.joinpath(f"{img_plane:05d}" + "_" + str(class_value) + ".tif")

            out_img = img[..., idx]

            skio.imsave(str(out_fpath_class), out_img, plugin="pil")
            # with tifffile.TiffWriter(str(out_fpath_class)) as stack:
            #     for img_plane in out_img:
            #         stack.save(img_plane)



if __name__ == "__main__":
    main()
