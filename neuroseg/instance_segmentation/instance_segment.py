from typing import Union, Tuple
import logging
from pathlib import Path
import os
import time
import multiprocessing
from functools import partial
import uuid

import numpy as np
import pandas as pd
import skimage.segmentation as skseg
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.color as skcolor
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import pyclesperanto_prototype as cle
import tifffile
from tqdm import tqdm
import pyvista as pv
import pymeshfix


from neuroseg.config import TrainConfig, PredictConfig
from neuroseg.utils import save_volume
from neuroseg.utils import IntegerShearingCorrect
from neuroseg.utils import stats
from neuroseg.utils import get_bbox
from neuroseg.tiledpredict.datapredictorbase import DataPredictorBase

_shared_voronoi_stack = None

class VoronoiInstanceSegmenter:
    def __init__(
        self,
        predicted_data: dict,
        config: Union[TrainConfig, PredictConfig] = None,
        output_path: Path = None,
        shearing_correct_delta: int = -7,
        block_size: int = None,
        downscaling_xy_factor: int = 4,
        padding_slices: int = 10,
        autocrop: bool = True,
        compute_meshes: bool = True
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

            self.enable_instance_segmentation = config.enable_instance_segmentation
            self.shearing_correct_delta = (
                config.instance_segmentation_shearing_correct_delta
            )
            self.block_size = config.instance_segmentation_block_size
            self.output_path = self.config.output_path
            self.downscaling_xy_factor = (
                self.config.instance_segmentation_downscaling_xy_factor
            )
            self.padding_slices = self.config.instance_segmentation_padding_slices
            self.autocrop = self.config.instance_segmentation_autocrop
            # TODO: add compute_meshes to config
            self.compute_meshes = True
        else:
            self.enable_instance_segmentation = True
            self.shearing_correct_delta = shearing_correct_delta
            self.output_path = output_path
            self.downscaling_xy_factor = downscaling_xy_factor
            self.padding_slices = padding_slices
            self.autocrop = autocrop
            self.compute_meshes = compute_meshes

            self.block_size = block_size

        if self.enable_instance_segmentation:
            devices = cle.available_device_names()
            device = devices[0]
            print("Using device:", device)
            self.predicted_data_dict = predicted_data

            # print("Performing instance segmentation...")
            for key, value in self.predicted_data_dict.items():

                imgname = Path(key).stem
                fname = f"{imgname}_segmented.tif"
                out_fpath = self.output_path.joinpath(fname)

                if out_fpath.exists():
                    # remove it
                    os.remove(str(out_fpath))
                    # binarized_path = out_fpath.joinpath(out_fpath.stem+"_binarized.tif")
                    # os.remove(str(binarized_path))

                if self.block_size is not None:
                    self.block_voronoi_segment(
                        value,
                        block_size=self.block_size,
                        out_fpath=out_fpath,
                        downscaling_xy_factor=downscaling_xy_factor,
                        padding_slices=padding_slices,
                        autocrop=self.autocrop,
                        get_stats=True,
                        get_meshes=True
                    )
                else:
                    segmented_volume = self.voronoi_segment(
                        np.squeeze(value), padding_slices=padding_slices
                    )
                    self.save_block(out_fpath, segmented_volume.astype(np.uint16))

    @staticmethod
    def pad_img(img: np.ndarray, padding_slices: int = 0):
        """Apply z padding"""
        if padding_slices > 0:
            lower_pad = img[:padding_slices][::-1]
            upper_pad = img[-padding_slices:][::-1]
            return np.concatenate([lower_pad, img, upper_pad], axis=0)
        else:
            return img

    @staticmethod
    def unpad_img(img: np.ndarray, padding_slices: int = 0):
        """Remove z padding from an image."""
        if padding_slices > 0:
            return img[padding_slices:-padding_slices]
        else:
            return img

    @classmethod
    def block_voronoi_segment(
        cls,
        input_img: np.ndarray,
        out_fpath: Path,
        shearing_correct_delta: int = -7,
        downscaling_xy_factor: int = 4,
        spot_detection_sigma: float = 3.0,
        outline_sigma: float = 1.0,
        threhsold_otsu: bool = True,
        threshold: float = 0.5,
        block_size: int = 100,
        padding_slices: int = 0,
        autocrop: bool = True,
        get_stats: bool = True,
        get_meshes: bool = True,
        n_meshing_threads: int = None,
        enable_meshing_multiprocessing: bool = True

    ) -> Tuple[np.ndarray, np.ndarray]:
        """ "
        Perform block-based Voronoi Segmentation using PyCLEsperanto.
        """
        n_imgs = input_img.shape[0]
        idx_batches = cls.split(list(range(n_imgs)), block_size)
        n_objects = 0
        voronoi_out = None
        last_img = None
        global_stats = pd.DataFrame()
        for idx, batch_idxs in enumerate(tqdm(idx_batches)):
            print(f"Voronoi prediction on batch {idx}/{len(idx_batches)}")
            sub_img = np.squeeze(input_img[batch_idxs[0] : batch_idxs[-1] + 1])
            if sub_img.shape == 2:
                # in this case we have a one-image stack
                sub_img = np.expand_dims(sub_img, 0)
            if autocrop:
                autocrop_range = DataPredictorBase._get_autocrop_range(sub_img)
                pre_crop_sub_img_shape = sub_img.shape
                sub_img = sub_img[:, :, autocrop_range[0] : autocrop_range[1]]
                autocrop_pads = (
                    (0, 0),
                    (0, 0),
                    (autocrop_range[0], pre_crop_sub_img_shape[2] - autocrop_range[1]),
                )
            
            pre_shearing_correct_shape = sub_img.shape

            # apply forward shearing correction
            shearing_correct = IntegerShearingCorrect(delta=shearing_correct_delta)
            sub_img, _ = shearing_correct.forward_correct(arr=sub_img)
            # performing voronoi segmentation
            voronoi_segmented = cls.voronoi_segment(
                sub_img,
                shearing_correct_delta=shearing_correct_delta,
                downscaling_xy_factor=downscaling_xy_factor,
                spot_detection_sigma=spot_detection_sigma,
                outline_sigma=outline_sigma,
                threhsold_otsu=threhsold_otsu,
                threshold=threshold,
                enable_shearing_correction=False,
                padding_slices=padding_slices,
            )

            voronoi_segmented = np.where(
                voronoi_segmented != 0, voronoi_segmented + n_objects, 0
            )
            
            n_objects = np.max(voronoi_segmented)

            # applying back shearing correction
            if last_img is not None:
                label_remap = {}
                new_block_pads = [(0, 0), (0, np.abs(shearing_correct_delta))]
                last_block_pads = [(0, 0), (np.abs(shearing_correct_delta), 0)]
                
                # take first match of the new block
                first_img_new_block = voronoi_segmented[0][:, :pre_crop_sub_img_shape[2]]
                # pad to fit last image of the last block
                first_img_new_block = np.pad(voronoi_segmented[0], new_block_pads)
                if autocrop:
                    first_img_new_block = np.pad(first_img_new_block, autocrop_pads[1:])
                # pad the last image of the last block
                last_img = np.pad(last_img, last_block_pads)
                # check overlap
                overlap = np.logical_and(last_img, first_img_new_block)

                labels_last_block = np.unique(last_img)
                # remove background
                labels_last_block = np.delete(
                    labels_last_block, np.where(labels_last_block == 0)
                )

                for label in labels_last_block:
                    label_mask = last_img == label
                    overlap = np.logical_and(label_mask, first_img_new_block)
                    overlap_labels_img = np.where(overlap, first_img_new_block, 0)

                    overlapping_labels = np.unique(overlap_labels_img)
                    overlapping_labels = np.delete(
                        overlapping_labels, np.where(overlapping_labels == 0)
                    )
                    if len(overlapping_labels) > 1:
                        to_remap = cls.get_largest_label(
                            overlap_labels_img, overlapping_labels
                        )
                        label_remap[to_remap] = label
                    elif len(overlapping_labels) == 1:
                        label_remap[overlapping_labels[0]] = label
                    else:
                        pass

                if label_remap != {}:
                    # fastest way I could find to replace values in a numpy array
                    # there probably is a better way somewhere
                    v = np.array(list(label_remap.keys()))
                    k = np.array(list(label_remap.values()))

                    sidx = k.argsort()
                    k = k[sidx]
                    v = v[sidx]

                    idx_search = np.searchsorted(k, voronoi_segmented.ravel()).reshape(
                        voronoi_segmented.shape
                    )
                    idx_search[idx_search == len(k)] = 0
                    mask = k[idx_search] == voronoi_segmented
                    voronoi_segmented = np.where(mask, v[idx_search], voronoi_segmented)
            
            # assigning last image
            last_img = voronoi_segmented[-1]
            last_img = last_img[:,-pre_crop_sub_img_shape[2]:]

            if get_stats:
                stats = pd.DataFrame(cle.statistics_of_labelled_pixels(sub_img, voronoi_segmented))
                stats.dropna(inplace=True)
            
            if get_meshes:
                stats_meshes = stats.copy()
                stats_meshes = stats_meshes.loc[stats_meshes.bbox_depth > 2]
                stats_meshes = stats_meshes.loc[stats_meshes.bbox_width > 4]
                stats_meshes = stats_meshes.loc[stats_meshes.bbox_height > 4]

                mesh_offset = np.array((batch_idxs[0], 0, 0))
                _shared_voronoi_stack = voronoi_segmented.copy()
                gen_mesh_partial = partial(cls.gen_mesh_skimage,
                    padding=2,
                    stats=stats,
                    voronoi_label_img=_shared_voronoi_stack,
                    mesh_offset=mesh_offset)

                if n_meshing_threads is None:
                    n_meshing_threads = multiprocessing.cpu_count()
                
                results = []
                mesh_res = []

                if enable_meshing_multiprocessing:
                    pool = multiprocessing.Pool(n_meshing_threads)
                    
                    for label, mesh in tqdm(pool.imap_unordered(gen_mesh_partial, stats.label.values, chunksize=100), total=len(stats)):
                        results.append((label, mesh))
                    
                    mesh_res = [(item[0], item[1]) for item in results if type(item[1]) == pv.PolyData]
                    pool.close()
                    pool.join()
                else:
                    for df_idx, neuron in tqdm(stats.iterrows(), total=len(stats)):
                        label, neuron_mesh = gen_mesh_partial(neuron.label)
                        if type(neuron_mesh) == pv.PolyData:
                            mesh_res.append((label, neuron_mesh))                        
                    
                # neuron_multiblock = pv.MultiBlock()

                # for neuron_mesh in mesh_res:
                #     neuron_multiblock.append(neuron_mesh)

                # save_pool = multiprocessing.Pool(n_meshing_threads)
                
                save_mesh_partial = partial(cls.save_mesh, out_fpath=out_fpath, idx=idx)

                saved_meshes = []

                for mesh_idx, mesh_res_tuple in enumerate(mesh_res):
                    label, fpath = save_mesh_partial(mesh_res_tuple, mesh_idx)
                    saved_meshes.append((label, fpath))
                # for saved_path_tuple in tqdm(save_pool.starmap(save_mesh_partial, zip(mesh_res, range(len(mesh_res))), chunksize=1000), total=len(mesh_res)):
                #        saved_meshes.append(saved_path_tuple)

                mesh_df = pd.DataFrame(saved_meshes)
                mesh_df.rename(columns={0: "label", 1: "mesh_path"}, inplace=True)
                stats = stats.merge(mesh_df, on="label", how="left")

                for label in stats.label.unique():
                    stats.loc[stats.label == label, 'UUID'] = uuid.uuid4()
                
                stats_cpy = stats.copy(deep=True)
                
                global_stats =  pd.concat([global_stats, stats_cpy])


                # saved_paths = save_pool.starmap(save_mesh, zip(mesh_res, range(len(mesh_res))))

                # multiblock_out_fpath = out_fpath.parent.joinpath(out_fpath.stem + f"_{idx}" + ".vtm")
                # print("saving multiblock")
                # neuron_multiblock.save(multiblock_out_fpath)
                # print("multiblock saved")

            if autocrop:
                voronoi_segmented = np.pad(voronoi_segmented, autocrop_pads)
                last_img = np.pad(last_img, autocrop_pads[1:])
            
            # reversing shearing correction
            voronoi_segmented = shearing_correct.inverse_correct(voronoi_segmented)

            cls.save_block(
                out_path=out_fpath, block=voronoi_segmented.astype(np.uint16)
            )

        stats_df_out = out_fpath.parent.joinpath(out_fpath.stem+ "_stats.csv")
        global_stats.to_csv(str(stats_df_out))
        # return voronoi_out
    @staticmethod
    def save_mesh(mesh_tuple: Tuple[int, pv.PolyData], mesh_idx: int, out_fpath: Path, idx: int):
        label, mesh = mesh_tuple
        mesh_out_dir_path = out_fpath.parent.joinpath("meshes")
        if not mesh_out_dir_path.exists():
            mesh_out_dir_path.mkdir(exist_ok=True)
        mesh_out_fpath = mesh_out_dir_path.joinpath(out_fpath.stem + f"_{idx}_{mesh_idx}" + ".ply")
        with pv.VtkErrorCatcher(raise_errors=True) as error_catcher:
            try:
                mesh.save(mesh_out_fpath)
            except RuntimeError as e:
                raise e
        return label, mesh_out_fpath


    @staticmethod
    def gen_mesh_skimage(
        label: int,
        voronoi_label_img: np.ndarray,
        stats: pd.DataFrame,
        padding: int = 2,
        mesh_offset: np.ndarray = np.array([0,0,0])
        ) -> Tuple[int, pv.PolyData]:
        label_bbox = get_bbox(cle_df=stats, neuron_label=label, return_vols=False)
        origin = np.array((
            label_bbox[0][0]-padding,
            label_bbox[1][0]-padding,
            label_bbox[2][0]-padding))
        if mesh_offset is not None:
            origin = origin + mesh_offset

        # extract larger img
        img = voronoi_label_img[
                label_bbox[0][0]-padding:label_bbox[0][1]+padding,
                label_bbox[1][0]-padding:label_bbox[1][1]+padding,
                label_bbox[2][0]-padding:label_bbox[2][1]+padding
            ].astype(np.uint8)

        img = np.where(img>0, 1, 0).astype(np.uint8)

        if any(np.array(img.shape) < 2):
            return label, -1

        try:
            verts, faces_3, normals, values = skmeas.marching_cubes(img)
        except RuntimeError as e:
            return label, -1
        arr3 = np.ones(faces_3.shape[0], dtype=faces_3.dtype) * 3
        arr3 = np.expand_dims(arr3, axis=-1)

        faces_4 = np.append(arr3, faces_3, axis=1)

        mesh = pv.PolyData(verts, faces_4)

        if not mesh.is_manifold:
            fixer = pymeshfix.MeshFix(mesh)
            fixer.repair()
            mesh = fixer.mesh

            if not mesh.is_manifold:
                print("Mesh is not manifold after fixing")

        mesh.translate(origin, inplace=True)

        
        return label, mesh

    
    @staticmethod
    def get_largest_label(img, uniques=None):
        if uniques is None:
            uniques = np.unique(img)

        counts = []
        uniques = np.delete(uniques, np.where(uniques == 0))
        for label in uniques:
            counts.append(np.count_nonzero(img == label))

        return uniques[np.argmax(counts)]

    @staticmethod
    def split(idx_list, batch_size):
        """Split a list of indices into batches of size batch_size"""
        split_ls = [
            idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)
        ]

        # we don't want an element of the last element of the list to feel lonely
        if len(split_ls[-1]) == 1:
            elem = split_ls[-2][-1]
            split_ls[-2] = split_ls[-2][:-1]
            split_ls[-1] = [elem] + split_ls[-1]

        return split_ls

    @staticmethod
    def save_block(out_path: Path, block: np.ndarray, save_binarized=True):
        """Save block to disk by appending to file"""
        with tifffile.TiffWriter(str(out_path), bigtiff=True, append=True) as tif:
            for img_plane in block:
                img_plane = np.expand_dims(img_plane, axis=-1)
                tif.write(img_plane, compression="zlib")

        if save_binarized:
            binarized_fpath = out_path.parent.joinpath(out_path.stem + "_binarized.tif")
            with tifffile.TiffWriter(
                str(binarized_fpath), bigtiff=True, append=True
            ) as tif:
                for img_plane in block:
                    img_plane = (np.expand_dims(img_plane, axis=-1) > 0).astype(
                        np.uint16
                    )
                    tif.write(img_plane, compression="zlib")

    @classmethod
    def voronoi_segment(
        cls,
        input_img: np.ndarray,
        shearing_correct_delta: int = -7,
        downscaling_xy_factor: int = 4,
        spot_detection_sigma: float = 3.0,
        outline_sigma: float = 1.0,
        threhsold_otsu: bool = True,
        threshold: float = 0.5,
        padding_slices: int = 0,
        enable_shearing_correction: bool = True
    ) -> np.ndarray:
        """
        Perform voronoi segmentation using PyCLEsperanto
        """
        if enable_shearing_correction:
            # correcting shearing distorsion
            shearing_correct = IntegerShearingCorrect(delta=shearing_correct_delta)
            corrected_img, _ = shearing_correct.forward_correct(arr=input_img)
        else:
            corrected_img = input_img

        # applying padding to avoid voronoi z-border effects
        corrected_img = cls.pad_img(corrected_img, padding_slices)
        corrected_img_gpu = cle.push(corrected_img)

        # downscaling on xy: we can find the maxima in a downscaled version of the tensor
        # to be used as Voronoi seeds in the full-res image
        if downscaling_xy_factor in [1, None]:
            reduced = corrected_img
            reduced_gpu = corrected_img_gpu
        else:
            reduced = skmeas.block_reduce(
                corrected_img, (1, downscaling_xy_factor, downscaling_xy_factor), np.max
            )
            reduced_gpu = cle.push(reduced)

        # blur the image to make maxima detection less noisy
        blurred_detection_reduced = cle.gaussian_blur(
            reduced_gpu,
            sigma_x=spot_detection_sigma,
            sigma_y=spot_detection_sigma,
            sigma_z=spot_detection_sigma,
        )

        # maxima detection
        detected_spots_reduced = cle.detect_maxima_box(
            blurred_detection_reduced, radius_x=0, radius_y=0, radius_z=0
        )

        # performing another blur on the image, this time with a lower sigma, to find contours
        blurred_outline_reduced = cle.gaussian_blur(
            reduced_gpu,
            sigma_x=outline_sigma,
            sigma_y=outline_sigma,
            sigma_z=outline_sigma,
        )

        # threhsolding the image
        if not threhsold_otsu:  # define manually the threshold val
            max_int = np.iinfo(blurred_outline_reduced.dtype).max
            binary_reduced = cle.threshold(
                blurred_outline_reduced, constant=max_int * threshold
            )
        else:  # or find it using Otsu
            binary_reduced = cle.threshold_otsu(blurred_outline_reduced)

        # the only maxima we're interested in are in the neuron contours
        selected_spots_reduced = cle.binary_and(binary_reduced, detected_spots_reduced)

        # upscaling the maxima to full-res
        if downscaling_xy_factor in [1, None]:
            selected_spots = selected_spots_reduced
        else:
            selected_spots = np.zeros_like(corrected_img)
            selected_spots[
                :, ::downscaling_xy_factor, ::downscaling_xy_factor
            ] = selected_spots_reduced

        # blurring the full-res image to find contours
        blurred_outline = cle.gaussian_blur(
            corrected_img_gpu,
            sigma_x=outline_sigma,
            sigma_y=outline_sigma,
            sigma_z=outline_sigma / 4,
        )

        # again threshold, either manual or otsu
        if not threhsold_otsu:
            max_int = np.iinfo(blurred_outline.dtype).max
            binary = cle.threshold(blurred_outline, constant=max_int * threshold)
        else:
            binary = cle.threshold_otsu(blurred_outline)

        # compiling the full-res Voronoi masked diagram
        voronoi_diagram = np.array(cle.masked_voronoi_labeling(selected_spots, binary))

        # removing the pads we applied
        voronoi_diagram = cls.unpad_img(voronoi_diagram, padding_slices)

        if enable_shearing_correction:
            # applying reverse shearing distortion correction
            voronoi_diagram = shearing_correct.inverse_correct(arr=voronoi_diagram)

        return voronoi_diagram


class WatershedInstanceSegmenter:
    def __init__(
        self,
        config: Union[TrainConfig, PredictConfig],
        predicted_data: dict,
    ):
        if config is not None:
            if config.config_type == "training":
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
            self.distance_transform_sampling = int(
                self.config.instance_segmentation_distance_transform_sampling
            )
            self.dist_thresh = float(
                self.config.instance_segmentation_distance_transform_threshold
            )
            self.watershed_line = bool(self.config.instance_segmentation_watershed_line)
            self.bg_level = int(self.config.instance_segmentation_bg_level)
            self.segmented_data_dict = {}
            self.segmented_data_rgb_dict = {}

            print("Performing instance segmentation...")
            for key, value in self.predicted_data_dict.items():
                segmented_volume, segmented_volume_rgb = self.instance_segment_img(
                    input_img=np.squeeze(value)
                )
                self.segmented_data_dict[key] = segmented_volume
                self.segmented_data_rgb_dict[key] = segmented_volume_rgb
                img_name = key.split(".")[0]
                self.save_segmentation(
                    segmented_volume=segmented_volume,
                    segmented_volume_rgb=segmented_volume_rgb,
                    img_name=img_name,
                )

    def instance_segment_img(
        self, input_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        logging.info("Segmenting instances...")

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
        dist_transform = distance_transform_edt(
            img, sampling=self.distance_transform_sampling
        )
        thresh = self.dist_thresh * np.max(dist_transform)
        sure_fg = np.where(dist_transform > self.dist_thresh, 255, 0)

        # Mark the unknown pixels as uncertain
        unknown = np.subtract(sure_bg, sure_fg)

        # Create markers
        markers = skmeas.label(sure_fg)
        # shift the markers so that the background is bg_level
        markers = markers + self.bg_level

        # set the background marker to 0
        markers[unknown == 255] = 0

        # apply watershed
        logging.info("Applying watershed...")
        labels = skseg.watershed(
            input_img, markers=markers, watershed_line=self.watershed_line
        )

        rgb_labels = skcolor.label2rgb(
            labels, bg_label=self.bg_level, bg_color=(0, 0, 0)
        )

        return labels, rgb_labels

    def save_segmentation(
        self,
        segmented_volume: np.ndarray,
        segmented_volume_rgb: np.ndarray,
        img_name: str = "segmented_vol",
    ):

        output_path = self.config.output_path
        logging.info(f"Saving segmentation to {output_path}")
        save_volume(
            segmented_volume, output_path, fname=f"{img_name}_instance_segmentation"
        )
        save_volume(
            segmented_volume_rgb,
            output_path,
            fname=f"{img_name}_instance_segmentation_rgb",
        )
