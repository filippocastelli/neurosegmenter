from argparse import ArgumentParser
from pathlib import Path

from neuroseg import PredictConfig
from neuroseg.instance_segmentation import VoronoiInstanceSegmenter
from skimage import io as skio
from zetastitcher import InputFile
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_path",
        type=str,
        required=True,
        dest="output_path",
        help="output_path")

    parser.add_argument("-i", "--input_path",
        type=str,
        required=True,
        dest="input_file",
        help="input_file")

    args = parser.parse_args()
    output_path = Path(args.output_path)
    input_file = Path(args.input_file)
    
    #in_img = InputFile(str(input_file))
    #in_img = skio.imread(str(input_file), plugin="pil")
    #in_img = np.random.randint(0, 255, size=(200, 200, 200), dtype=np.uint8)
    in_img = InputFile(input_file)

    #skio.imsave(output_path.joinpath(input_file.stem+"sub.tif"),np.expand_dims(in_img, axis=0))
    data = {input_file.name: in_img}
    seg = VoronoiInstanceSegmenter(
        predicted_data=data,
        output_path=output_path,
        shearing_correct_delta=-7,
        block_size=60,
        downscaling_xy_factor=None,
        padding_slices=5
        )

if __name__ == "__main__":
    main()