from typing import Union, Tuple
import logging

import numpy as np
import skimage.segmentation as skseg
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.color as skcolor
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt

from neuroseg.config import TrainConfig, PredictConfig
from neuroseg.utils import save_volume


class InstanceSegmenter:

    def __init__(self,
                 config: Union[TrainConfig, PredictConfig],
                 predicted_data: dict):
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






