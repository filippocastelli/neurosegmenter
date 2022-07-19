import pyclesperanto_prototype as cle
import pandas as pd
import numpy as np


def get_bbox(
    neuron_label: int,
    cle_df: pd.DataFrame,
    stack_img: np.ndarray = None,
    stack_label: np.ndarray = None,
    return_vols: bool = True,
    binary_label: bool = True,
    extra_xy: int = 0,
    extra_z: int = 0):
    """
    Returns the bounding box of a neuron in the dataframe
    """

    df_row = cle_df.loc[cle_df.label == neuron_label]
    bbox_min_x = df_row.bbox_min_x.values[0].astype(int) - extra_xy
    bbox_min_y = df_row.bbox_min_y.values[0].astype(int) - extra_xy
    bbox_max_x = df_row.bbox_max_x.values[0].astype(int) + extra_xy
    bbox_max_y = df_row.bbox_max_y.values[0].astype(int) + extra_xy
    bbox_min_z = df_row.bbox_min_z.values[0].astype(int) - extra_z
    bbox_max_z = df_row.bbox_max_z.values[0].astype(int) + extra_z

    centroid_x = df_row.centroid_x.values[0].astype(int)
    centroid_y = df_row.centroid_y.values[0].astype(int)
    centroid_z = df_row.centroid_z.values[0].astype(int)

    middle_z = (bbox_min_z + bbox_max_z) // 2 - bbox_min_z

    if not return_vols:
        return (
            (bbox_min_z, bbox_max_z),
            (bbox_min_y, bbox_max_y),
            (bbox_min_x, bbox_max_x)
        )
    neuron_img = stack_img[bbox_min_z:bbox_max_z, bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]
    neuron_label_img = stack_label[bbox_min_z:bbox_max_z, bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]
    
    if binary_label:
        neuron_label_img = np.where(neuron_label_img == neuron_label, 1, 0)

    return neuron_img, neuron_label_img