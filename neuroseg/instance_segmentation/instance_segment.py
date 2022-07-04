from typing import Union, Tuple
import logging
from pathlib import Path
import os
import time

import numpy as np
import skimage.segmentation as skseg
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.color as skcolor
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import pyclesperanto_prototype as cle
import tifffile
from tqdm import tqdm


from neuroseg.config import TrainConfig, PredictConfig
from neuroseg.utils import save_volume
from neuroseg.utils import IntegerShearingCorrect
from neuroseg.tiledpredict.datapredictorbase import DataPredictorBase

        
class VoronoiInstanceSegmenter:
    def __init__(self,
        predicted_data: dict,
        config: Union[TrainConfig, PredictConfig] = None,
        output_path: Path = None,
        shearing_correct_delta: int = -7,
        block_size: int = None,
        downscaling_xy_factor: int = 4,
        padding_slices: int = 10,
        autocrop: bool = True
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
            self.block_size = config.instance_segmentation_block_size
            self.output_path = self.config.output_path
            self.downscaling_xy_factor = self.config.instance_segmentation_downscaling_xy_factor
            self.padding_slices = self.config.instance_segmentation_padding_slices
            self.autocrop = self.config.instance_segmentation_autocrop
            #self.autocrop = autocrop # TODO: add autocrop config for VoronoiInstanceSegmenter
        else:
            self.enable_instance_segmentation = True
            self.shearing_correct_delta = shearing_correct_delta
            self.output_path = output_path
            self.downscaling_xy_factor = downscaling_xy_factor
            self.padding_slices = padding_slices
            self.autocrop = autocrop
            
            self.block_size = block_size
        devices = cle.available_device_names()
        device = devices[0]
        print("Using device:", device)

        if self.enable_instance_segmentation:
            self.predicted_data_dict = predicted_data

            #print("Performing instance segmentation...")
            for key, value in self.predicted_data_dict.items():

                imgname = Path(key).stem
                fname = f"{imgname}_segmented.tif"
                out_fpath = self.output_path.joinpath(fname)

                if out_fpath.exists():
                    # remove it
                    os.remove(str(out_fpath))
                    #binarized_path = out_fpath.joinpath(out_fpath.stem+"_binarized.tif")
                    #os.remove(str(binarized_path))


                if self.block_size is not None:
                    self.block_voronoi_segment(
                        value,
                        block_size=self.block_size,
                        out_fpath=out_fpath,
                        downscaling_xy_factor=downscaling_xy_factor,
                        padding_slices=padding_slices,
                        autocrop=self.autocrop)
                else:
                    segmented_volume = self.voronoi_segment(np.squeeze(value), padding_slices=padding_slices)
                    self.save_block(out_fpath, segmented_volume.astype(np.uint16))

                #self.predicted_data_dict[key] = segmented_volume

                # seg_volume_max = np.max(segmented_volume)
                # # save_volume(segmented_volume, self.config.output_path / f"{imgname}_segmented")
                # if seg_volume_max < 255:
                #     save_8bit, save_16bit, save_32bit = True, False, False
                # elif seg_volume_max < 65535:
                #     save_8bit, save_16bit, save_32bit = False, True, False
                # else:
                #     save_8bit, save_16bit, save_32bit = False, False, True
                # save_volume(segmented_volume,
                #     output_path=self.output_path,
                #     fname=f"{imgname}_segmented",
                #     save_tiff=False,
                #     save_8bit=save_8bit,
                #     save_16bit=save_16bit,
                #     save_32bit=save_32bit)


    @staticmethod
    def pad_img(img: np.ndarray, padding_slices: int = 0):
        if padding_slices > 0:
            lower_pad = img[:padding_slices][::-1]
            upper_pad = img[-padding_slices:][::-1]
            return np.concatenate([lower_pad, img, upper_pad], axis=0)
        else:
            return img
    
    @staticmethod
    def unpad_img(img: np.ndarray, padding_slices: int = 0):
        if padding_slices > 0:
            return img[padding_slices:-padding_slices]
        else:
            return img


    @classmethod
    def block_voronoi_segment(cls,
        input_img: np.ndarray,
        out_fpath: Path,
        shearing_correct_delta: int = -7,
        downscaling_xy_factor: int = 4,
        spot_detection_sigma: float = 3.,
        outline_sigma: float = 1.,
        threhsold_otsu: bool = True,
        threshold: float = 0.5,
        back_shearing_correct: bool = True,
        block_size: int = 100,
        padding_slices: int = 0,
        autocrop: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
        
        n_imgs = input_img.shape[0]
        idx_batches = cls.split(list(range(n_imgs)), block_size)
        n_objects = 0
        voronoi_out = None
        last_img = None
        for idx, batch_idxs in enumerate(tqdm(idx_batches)):
            print(f"Voronoi prediction on batch {idx}/{len(idx_batches)}")
            sub_img = np.squeeze(input_img[batch_idxs[0]:batch_idxs[-1]+1])

            if autocrop:
                autocrop_range = DataPredictorBase._get_autocrop_range(sub_img)
                pre_crop_sub_img_shape = sub_img.shape
                sub_img = sub_img[:, :, autocrop_range[0]:autocrop_range[1]]
                autocrop_pads = ((0,0), (0,0), (autocrop_range[0], pre_crop_sub_img_shape[2]-autocrop_range[1]))

            voronoi_segmented = cls.voronoi_segment(
                sub_img,
                shearing_correct_delta=shearing_correct_delta,
                downscaling_xy_factor=downscaling_xy_factor,
                spot_detection_sigma=spot_detection_sigma,
                outline_sigma=outline_sigma,
                threhsold_otsu=threhsold_otsu,
                threshold=threshold,
                back_shearing_correct=True,
                padding_slices=padding_slices,
            )
            voronoi_segmented = np.where(voronoi_segmented != 0, voronoi_segmented + n_objects, 0)
            n_objects = np.max(voronoi_segmented)
            if last_img is not None:
                label_remap = {}
                new_block_pads = [(0,0), (0,np.abs(shearing_correct_delta))]
                last_block_pads = [(0,0), (np.abs(shearing_correct_delta),0)]

                first_img_new_block = np.pad(voronoi_segmented[0], new_block_pads)
                if autocrop:
                    first_img_new_block = np.pad(first_img_new_block, autocrop_pads[1:])
                last_img = np.pad(last_img, last_block_pads)
                # check overlap
                overlap = np.logical_and(last_img, first_img_new_block)

                labels_last_block = np.unique(last_img)
                # remove background
                labels_last_block = np.delete(labels_last_block, np.where(labels_last_block==0))

                for label in labels_last_block:
                    label_mask = last_img == label
                    overlap = np.logical_and(label_mask, 
                        first_img_new_block)
                    overlap_labels_img = np.where(overlap, first_img_new_block, 0)

                    overlapping_labels = np.unique(overlap_labels_img)
                    overlapping_labels = np.delete(overlapping_labels, np.where(overlapping_labels == 0))
                    if len(overlapping_labels) > 1:
                        to_remap = cls.get_largest_label(overlap_labels_img, overlapping_labels)
                        label_remap[to_remap] = label
                    elif len(overlapping_labels) == 1:
                        label_remap[overlapping_labels[0]] = label
                    else:
                        pass

                # def mp(entry):
                #    return label_remap[entry] if entry in label_remap else entry
                # mp_vector = np.vectorize(mp)

                # voronoi_segmented_cpy = voronoi_segmented.copy()
                # st = time.time()
                # for to_remap, label in tqdm(label_remap.items()):
                #    voronoi_segmented_cpy[voronoi_segmented_cpy==label] = to_remap
                # end = time.time()
                # time_for = end - st

                if label_remap != {}:
                    v = np.array(list(label_remap.keys()))
                    k = np.array(list(label_remap.values()))

                    sidx = k.argsort()
                    k = k[sidx]
                    v = v[sidx]

                    idx = np.searchsorted(k, voronoi_segmented.ravel()).reshape(voronoi_segmented.shape)
                    idx[idx==len(k)]=0
                    mask = k[idx] == voronoi_segmented
                    voronoi_segmented = np.where(mask, v[idx], voronoi_segmented)

                # time_vec = time.time() - end

                # print(f"time for: {time_for}")
                # print(f"time vec: {time_vec}")
                #voronoi_segmented = mp_vector(voronoi_segmented)
            last_img = voronoi_segmented[-1]

            if autocrop:
                voronoi_segmented = np.pad(voronoi_segmented, autocrop_pads)
                last_img = np.pad(last_img, autocrop_pads[1:])

            cls.save_block(out_path=out_fpath, block=voronoi_segmented.astype(np.uint16))

        #return voronoi_out

    @staticmethod
    def get_largest_label(img, uniques=None):
        if uniques is None:
            uniques = np.unique(img)
        
        counts = []
        uniques = np.delete(uniques, np.where(uniques==0))
        for label in uniques:
            counts.append(np.count_nonzero(img==label))

        return uniques[np.argmax(counts)]
        



    @staticmethod
    def split(idx_list, batch_size):
        return [idx_list[i:i+batch_size] for i in range(0, len(idx_list), batch_size)]
    
    @staticmethod
    def save_block(out_path:Path, block:np.ndarray, save_binarized=True):
        with tifffile.TiffWriter(str(out_path), bigtiff=True, append=True) as tif:
            for img_plane in block:
                img_plane = np.expand_dims(img_plane, axis=-1)
                tif.write(img_plane, compression="zlib")

        if save_binarized:
            binarized_fpath = out_path.parent.joinpath(out_path.stem+"_binarized.tif")
            with tifffile.TiffWriter(str(binarized_fpath), bigtiff=True, append=True) as tif:
                for img_plane in block:
                    img_plane = (np.expand_dims(img_plane, axis=-1) > 0).astype(np.uint16)
                    tif.write(img_plane, compression="zlib")

    @classmethod
    def voronoi_segment(cls,
        input_img: np.ndarray,
        shearing_correct_delta: int = -7,
        downscaling_xy_factor: int = 4,
        spot_detection_sigma: float = 3.,
        outline_sigma: float = 1.,
        threhsold_otsu: bool = True,
        threshold: float = 0.5,
        back_shearing_correct: bool = True,
        padding_slices: int = 0,
        ) -> np.ndarray:
        """
        Perform voronoi segmentation using PyCLEsperanto
        """
        #print("Performing voronoi segmentation...")

        # correcting shearing distorsion

        shearing_correct = IntegerShearingCorrect(
            delta=shearing_correct_delta)

        corrected_img, data_mask = shearing_correct.forward_correct(arr=input_img)
        corrected_img = cls.pad_img(corrected_img, padding_slices)
        corrected_img_gpu = cle.push(corrected_img)

        # max downscaling
        if downscaling_xy_factor in [1, None]:
            reduced = corrected_img
            reduced_gpu = corrected_img_gpu
        else:
            reduced = skmeas.block_reduce(corrected_img, (1, downscaling_xy_factor, downscaling_xy_factor), np.max)
            reduced_gpu = cle.push(reduced)

        blurred_detection_reduced = cle.gaussian_blur(
            reduced_gpu,
            sigma_x=spot_detection_sigma,
            sigma_y=spot_detection_sigma,
            sigma_z=spot_detection_sigma)

        detected_spots_reduced = cle.detect_maxima_box(blurred_detection_reduced, radius_x=0, radius_y=0, radius_z=0)
        blurred_outline_reduced = cle.gaussian_blur(reduced_gpu, sigma_x=outline_sigma, sigma_y=outline_sigma, sigma_z=outline_sigma)
        if not threhsold_otsu:
            max_int = np.iinfo(blurred_outline_reduced.dtype).max
            binary_reduced = cle.threshold(blurred_outline_reduced, constant=max_int*threshold)
        else:
            binary_reduced = cle.threshold_otsu(blurred_outline_reduced)

        selected_spots_reduced = cle.binary_and(binary_reduced, detected_spots_reduced)
        
        # upscaling selected_spots
        if downscaling_xy_factor in [1, None]:
            selected_spots = selected_spots_reduced
        else:
            selected_spots = np.zeros_like(corrected_img)
            selected_spots[:, ::downscaling_xy_factor, ::downscaling_xy_factor] = selected_spots_reduced

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
        voronoi_diagram = cls.unpad_img(voronoi_diagram, padding_slices)

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







