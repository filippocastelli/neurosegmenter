from argparse import ArgumentParser
from pathlib import Path

from neuroseg import PredictConfig
from neuroseg.instance_segmentation import VoronoiInstanceSegmenter
from skimage import io as skio

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
    
    
    in_img = skio.imread(input_file, plugin="pil")
    data = {input_file.name: in_img}
    seg = VoronoiInstanceSegmenter(
        predicted_data=data,
        output_path=output_path,
        shearing_correct_delta=-7
        )