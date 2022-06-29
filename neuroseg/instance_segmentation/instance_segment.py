from typing import Union, Tuple
import logging
from pathlib import Path

import numpy as np
import skimage.segmentation as skseg
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.color as skcolor
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import pyclesperanto_prototype as cle

from neuroseg.config import TrainConfig, PredictConfig
from neuroseg.utils import save_volume
from neuroseg.utils import IntegerShearingCorrect

class VoronoiInstanceSegmenter:
    def __init__(self,
        predicted_data: dict,
        config: Union[TrainConfig, PredictConfig] = None,
        output_path: Path = None,
        shearing_correct_delta: int = -7,

        ):
        self.config = config
        if config is not None:
            if config.config_type == "training" and config is not None:
                data_mode = config.dataset_mode
            elif config.config_type == "predict" and config is not None:
                data_mode = config.data_mode
            else:
                raise NotImplementedError
            if data_mode == "zetastitcher":
                raise NotImplementedError("ZetaStitcher data mode not supported yet.")
                # TODO: implement chunk-based instance segmentation 

            self.enable_instance_segmentation = config.enable_instance_segmentation
            self.shearing_correct_delta = config.instance_segmentation_shearing_correct_delta
            self.output_path = self.config.output_path
        else:
            self.enable_instance_segmentation = True
            self.shearing_correct_delta = shearing_correct_delta
            self.output_path = output_path

        devices = cle.available_device_names()
        device = devices[0]
        print("Using device:", device)

        if self.enable_instance_segmentation:
            self.predicted_data_dict = predicted_data

            print("Performing instance segmentation...")
            for key, value in self.predicted_data_dict.items():
                segmented_volume = self.voronoi_segment(np.squeeze(value))
                self.predicted_data_dict[key] = segmented_volume
                imgname = Path(key).stem
                # save_volume(segmented_volume, self.config.output_path / f"{imgname}_segmented")
                if np.max(value) < 255:
                    save_8bit, save_16bit, save_32bit = True, False, False
                elif np.max(value) < 65535:
                    save_8bit, save_16bit, save_32bit = False, True, False
                else:
                    save_8bit, save_16bit, save_32bit = False, False, True
                save_volume(segmented_volume,
                    output_path=self.output_path,
                    fname=f"{imgname}_segmented",
                    save_tiff=False,
                    save_8bit=save_8bit,
                    save_16bit=save_16bit,
                    save_32bit=save_32bit)
    @staticmethod
    def voronoi_segment(
        input_img: np.ndarray,
        shearing_correct_delta: int = -7,
        downscaling_factor_xy: int = 4,
        spot_detection_sigma: float = 3.,
        outline_sigma: float = 1.,
        threhsold_otsu: bool = True,
        threshold: float = 0.5,
        back_shearing_correct: bool = True,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform voronoi segmentation using PyCLEsperanto
        """
        print("Performing voronoi segmentation...")

        # correcting shearing distorsion
        shearing_correct = IntegerShearingCorrect(
            delta=shearing_correct_delta)

        corrected_img, data_mask = shearing_correct.forward_correct(arr=input_img)
        corrected_img_gpu = cle.push(corrected_img)

        # max downscaling 
        reduced_max = skmeas.block_reduce(corrected_img, (1, downscaling_factor_xy, downscaling_factor_xy), np.max)
        reduced_max_gpu = cle.push(reduced_max)

        blurred_detection_reduced = cle.gaussian_blur(
            reduced_max_gpu,
            sigma_x=spot_detection_sigma,
            sigma_y=spot_detection_sigma,
            sigma_z=spot_detection_sigma)

        detected_spots_reduced = cle.detect_maxima_box(blurred_detection_reduced, radius_x=0, radius_y=0, radius_z=0)
        blurred_outline_reduced = cle.gaussian_blur(reduced_max_gpu, sigma_x=outline_sigma, sigma_y=outline_sigma, sigma_z=outline_sigma)
        if not threhsold_otsu:
            max_int = np.iinfo(blurred_outline_reduced.dtype).max
            binary_reduced = cle.threshold(blurred_outline_reduced, constant=max_int*threshold)
        else:
            binary_reduced = cle.threshold_otsu(blurred_outline_reduced)

        selected_spots_reduced = cle.binary_and(binary_reduced, detected_spots_reduced)
        
        # upscaling selected_spots
        selected_spots = np.zeros_like(corrected_img)
        selected_spots[:, ::downscaling_factor_xy, ::downscaling_factor_xy] = selected_spots_reduced

        blurred_outline = cle.gaussian_blur(
            corrected_img_gpu,
            sigma_x=outline_sigma,
            sigma_y=outline_sigma,
            sigma_z=outline_sigma/4)
        
        if not threhsold_otsu:
            max_int = np.iinfo(blurred_outline.dtype).max
            binary = cle.threshold(blurred_outline, constant=max_int*threshold)
        else:
            binary = cle.threshold_otsu(blurred_outline)

        voronoi_diagram = np.array(cle.masked_voronoi_labeling(selected_spots, binary))

        if back_shearing_correct:
            voronoi_diagram = shearing_correct.inverse_correct(arr=voronoi_diagram)

        return voronoi_diagram


    

class InstanceSegmenter:

    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 predicted_data: dict,):
        if config is not None: 
            if config.config_type == 'training':
                data_mode = config.dataset_mode
            elif config.config_type == "predict":
                data_mode = config.data_mode
            else:
                raise NotImplementedError(config.config_type)
            if data_mode == "zetastitcher":
                raise NotImplementedError("ZetaStitcher data mode not supported yet.")
                # TODO: implement chunk-based instance segmentation 

        self.config = config
        self.enable_instance_segmentation = self.config.enable_instance_segmentation

        if self.enable_instance_segmentation:
            self.predicted_data_dict = predicted_data

            self.kernel_size = self.config.instance_segmentation_kernel_size
            self.kernel_sem = np.ones(self.kernel_size, np.uint8)
            self.clear_borders = bool(self.config.instance_segmentation_clear_borders)
            self.distance_transform_sampling = int(self.config.instance_segmentation_distance_transform_sampling)
            self.dist_thresh = float(self.config.instance_segmentation_distance_transform_threshold)
            self.watershed_line = bool(self.config.instance_segmentation_watershed_line)
            self.bg_level = int(self.config.instance_segmentation_bg_level)
            self.segmented_data_dict = {}
            self.segmented_data_rgb_dict = {}

            print("Performing instance segmentation...")
            for key, value in self.predicted_data_dict.items():
                segmented_volume, segmented_volume_rgb = self.instance_segment_img(input_img=np.squeeze(value))
                self.segmented_data_dict[key] = segmented_volume
                self.segmented_data_rgb_dict[key] = segmented_volume_rgb
                img_name = key.split(".")[0]
                self.save_segmentation(
                    segmented_volume=segmented_volume,
                    segmented_volume_rgb=segmented_volume_rgb,
                    img_name=img_name)

    def instance_segment_img(self, input_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        logging.info('Segmenting instances...')

        logging.info("Applying morphological operations...")
        # Threshold the image
        threshold = threshold_otsu(input_img)
        img = np.where(input_img > threshold, 255, 0)

        # Remove small objects by applying an opening operation
        img = skmorph.opening(img, selem=self.kernel_sem)

        # Determine confident background pixels
        sure_bg = skmorph.dilation(img, selem=self.kernel_sem)

        # Determine confident foreground pixels
        # Calculate the distance transform
        dist_transform = distance_transform_edt(img, sampling=self.distance_transform_sampling)
        thresh = self.dist_thresh * np.max(dist_transform)
        sure_fg = np.where(dist_transform > self.dist_thresh, 255, 0)

        # Mark the unknown pixels as uncertain
        unknown = np.subtract(sure_bg, sure_fg)

        # Create markers
        markers = skmeas.label(sure_fg)
        # shift the markers so that the background is bg_level
        markers = markers+self.bg_level

        # set the background marker to 0
        markers[unknown == 255] = 0

        # apply watershed
        logging.info("Applying watershed...")
        labels = skseg.watershed(input_img, markers=markers, watershed_line=self.watershed_line)

        rgb_labels = skcolor.label2rgb(labels, bg_label=self.bg_level, bg_color=(0, 0, 0))

        return labels, rgb_labels

    def save_segmentation(self,
                          segmented_volume:np.ndarray,
                          segmented_volume_rgb: np.ndarray,
                          img_name: str = 'segmented_vol'):

        output_path = self.config.output_path
        logging.info(f'Saving segmentation to {output_path}')
        save_volume(segmented_volume, output_path, fname=f"{img_name}_instance_segmentation")
        save_volume(segmented_volume_rgb, output_path, fname=f"{img_name}_instance_segmentation_rgb")







