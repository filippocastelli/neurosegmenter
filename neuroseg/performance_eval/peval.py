from re import L
from typing import Union, List, Tuple, Callable
import uuid
import numpy as np
import sklearn.metrics as skmetrics
import logging
from pathlib import Path
import h5py
import csv
import pickle

from skimage import io as skio
import networkx as nx

from neuroseg.utils import load_volume, glob_imgs, save_volume
from neuroseg.config import TrainConfig, PredictConfig


class PerformanceMetrics:
    """general class for performance metrics evaluation"""

    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 thr: float,
                 enable_curves: bool = False):

        self.enable_curves = enable_curves
        self.y_true = np.copy(y_true)
        self.y_pred = np.copy(y_pred)
        # self.y_pred = np.copy(y_pred).astype(np.uint8) # AM I RETARDED
        self.thr = thr

        self.y_true_fuzzy = y_true
        self.y_pred_fuzzy = y_pred
        
        self.fuzzy_intersection = np.sum(self.y_pred_fuzzy.flatten() * self.y_true_fuzzy.flatten())
        self.fuzzy_summation = np.sum(self.y_pred_fuzzy.flatten()) + np.sum(self.y_true_fuzzy.flatten())
        self.fuzzy_union = self.fuzzy_summation - self.fuzzy_intersection

        # self.y_pred = self.threshold_array(self.y_pred, thr, to_bool=True)
        # self.y_true = self.threshold_array(self.y_true, thr, to_bool=True)
        self.y_true = self.y_true > thr
        self.y_pred = self.y_pred > thr

        self.tp, self.fp, self.tn, self.fn = self.cardinal_metrics()

        self.specificity = self.crisp_specificity()
        self.recall = self.crisp_recall()
        self.precision = self.crisp_precision()
        self.false_negative_ratio = self.crisp_false_negative_ratio()
        self.fallout = self.crisp_fallout()

        self.measure_dict = self.create_dict()

    def cardinal_metrics(self, sum_elems: bool = True) -> tuple:
        # TP
        true_positive = np.logical_and(self.y_pred, self.y_true)
        # TN
        true_negative = np.logical_and(
            np.logical_not(self.y_pred), np.logical_not(self.y_true)
        )
        # FP
        false_positive = np.logical_and(self.y_pred == True, self.y_true == False)
        # FN
        false_negative = np.logical_and(self.y_pred == False, self.y_true == True)

        if sum_elems:
            return (
                np.sum(true_positive),
                np.sum(false_positive),
                np.sum(true_negative),
                np.sum(false_negative),
            )
        else:
            return true_positive, false_positive, true_negative, false_negative

    def fuzzy_dice(self) -> float:
        """Dice coefficient for fuzzy segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (6)
        """
        return 2 * self.fuzzy_intersection / self.fuzzy_summation

    def fuzzy_jaccard(self) -> float:
        """Jaccard index for fuzzy segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (7)
        """
        return self.fuzzy_intersection / self.fuzzy_union

    @staticmethod
    def threshold_array(arr: np.ndarray,
                        thr: float = 0.5,
                        to_bool: bool = False) -> np.ndarray:
        arr[arr >= thr] = 1
        arr[arr < thr] = 0

        if to_bool:
            return np.array(arr, dtype=bool)
        else:
            return arr

    def crisp_jaccard(self) -> float:
        """Jaccard index for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (7)
        """
        return self.tp / (self.tp + self.fp + self.fn)

    def crisp_dice(self) -> float:
        """Dice coefficient for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (6)
        """
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    # recall/sensitivity
    def crisp_recall(self) -> float:
        """Recall or Sensitivity or TPR for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (10)
        """
        return self.tp / (self.tp + self.fn)

    # specificity/true negative ratio
    def crisp_specificity(self) -> float:
        """Specificity or TNR for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (11)
        """
        return self.tn / (self.tn + self.fp)

    # fallout/false positive ratio
    def crisp_fallout(self) -> float:
        """Fallout or FPR for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (12)
        """
        return 1 - self.specificity

    def crisp_false_negative_ratio(self) -> float:
        """FNR for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (13)
        """
        return 1 - self.recall

    # precision/positive predictive ratio
    def crisp_precision(self) -> float:
        """Precision or PPV for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (14)
        """
        return self.tp / (self.tp + self.fp)

    def crisp_auc(self) -> float:
        """Estimator of Area under ROC
        ROC is defined as TPR vs FPR plot, AUC here is calculated as area of the
        trapezoid defined by the measurement opint of the lines TPR=0 nad FPR=1.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (47)
        """
        return 1 - ((self.fallout + self.false_negative_ratio) / 2)

    def crisp_f1_score(self) -> float:
        """F1-score or FMS1 for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (16)
        """
        return (2 * self.precision * self.recall) / (self.precision + self.recall)

    def sk_f1_score(self) -> float:
        """F1-score implemented by scikit-learn"""
        return skmetrics.f1_score(self.y_true.flatten(), self.y_pred.flatten())

    def sk_roc_curve(self) -> dict:
        """ROC curve
        Implemented by scikit-learn"""
        logging.info("Calculating ROC curve")
        fpr, tpr, thresholds = skmetrics.roc_curve(y_true=self.y_true_fuzzy.flatten(),
                                                   y_score=self.y_pred_fuzzy.flatten())

        roc_dict = {"fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds}
        return roc_dict

    def sk_roc_auc_score(self) -> float:
        """Area under ROC curve
        Implemented by scikit-learn"""
        logging.info("Calculating PR curve")
        flatten_y_true = self.y_true.flatten()
        flatten_y_pred = self.y_pred.flatten()

        return skmetrics.roc_auc_score(y_true=flatten_y_true,
                                       y_score=flatten_y_pred)

    def sk_pr_curve(self) -> dict:
        """Precision Recall curve
        Implemented by scikit-learn"""
        precision, recall, threhsolds = skmetrics.precision_recall_curve(y_true=self.y_true_fuzzy.flatten(),
                                                                         probas_pred=self.y_pred_fuzzy.flatten())

        pr_dict = {"precision": precision,
                   "recall": recall,
                   "thresholds": threhsolds}
        return pr_dict

    def create_dict(self) -> dict:
        crisp_jaccard = self.crisp_jaccard()
        crisp_dice = self.crisp_dice()
        crisp_recall = self.crisp_recall()
        crisp_specificity = self.crisp_specificity()
        crisp_fallout = self.crisp_fallout()
        crisp_false_negative_ratio = self.crisp_false_negative_ratio()
        crisp_precision = self.crisp_precision()
        crisp_auc = self.crisp_auc()
        crisp_f1_score = self.crisp_f1_score()
        sk_f1_score = self.sk_f1_score()
        fuzzy_dice = self.fuzzy_dice()
        fuzzy_jaccard = self.fuzzy_jaccard()

        sk_roc_auc_score = self.sk_roc_auc_score()

        dictionary = {
            "crisp_jaccard": crisp_jaccard,
            "crisp_dice": crisp_dice,
            "crisp_recall": crisp_recall,
            "crisp_specificity": crisp_specificity,
            "crisp_fallout": crisp_fallout,
            "crisp_false_negative_ratio": crisp_false_negative_ratio,
            "crisp_precision": crisp_precision,
            "crisp_auc": crisp_auc,
            "crisp_f1_score": crisp_f1_score,
            "sk_f1_score": sk_f1_score,
            "fuzzy_dice": fuzzy_dice,
            "fuzzy_jaccard": fuzzy_jaccard,
            "sk_roc_auc_score": sk_roc_auc_score
        }

        if self.enable_curves:
            roc_curve = self.sk_roc_curve()
            pr_curve = self.sk_pr_curve()

            dictionary.update({
                "roc_curve": roc_curve,
                "pr_curve": pr_curve})

        return dictionary

    def get_metrics(self) -> dict:
        return self.measure_dict


class SingleVolumePerformanceEvaluator:

    def __init__(self,
                 config: Union[PredictConfig, TrainConfig],
                 predicted_data: np.ndarray = None,
                 gt_array: np.ndarray = None):
        self.config = config

        self.n_output_classes = config.n_output_classes
        # self.class_values = config.class_values

        self.soft_labels = config.soft_labels
        if config.config_type == "predict":
            self.normalize_ground_truth = config.ground_truth_normalize
        elif config.config_type == "training":
            self.normalize_ground_truth = config.normalize_masks

        if gt_array is None:
            self.ground_truth_mode = config.ground_truth_mode
            self.ground_truth_path = self._get_gt_path()
            self._load_gt()
        else:
            self.ground_truth = gt_array

        self._preprocess_pred(predicted_data)

        self.classification_threshold = config.pe_classification_threshold
        self.enable_curves = config.pe_enable_curves
        self._calc_metrics()

    def _get_gt_path(self) -> Union[Path, list]:
        if self.ground_truth_mode == "stack":
            if self.config.ground_truth_path.is_file():
                return self.config.ground_truth_path
            elif self.config.ground_truth_path.is_dir():
                return glob_imgs(self.config.ground_truth_path, mode="stack", to_string=False)[0]
            else:
                raise ValueError(f"invalid ground truth path {str(self.config.ground_truth_path)}")
        elif self.ground_truth_mode == "single_images":
            return self.config.ground_truth_path
        else:
            raise NotImplementedError(self.ground_truth_mode)

    def _load_data(self, pred_array):
        self._load_gt()
        # self._load_predictions(pred_array)

    def _preprocess_pred(self, pred_array: np.ndarray) -> None:
        if len(pred_array.shape) > len(self.ground_truth.shape):
            pred_array = np.squeeze(pred_array)
        elif len(pred_array.shape) < len(self.ground_truth.shape):
            pred_array = np.expand_dims(pred_array, axis=-1)
        else:
            pass

        assert (pred_array.shape == self.ground_truth.shape), "GT and predictions have different number of channels"

        self.predictions = pred_array

    def _load_gt(self) -> None:
        gt_vol, norm = load_volume(self.ground_truth_path,
                                   ignore_last_channel=False,
                                   data_mode=self.ground_truth_mode,
                                   return_norm=True)

        if self.n_output_classes > 1:
            class_val_img_list = []
            for output_class_value in self.class_values:
                class_val_img_list.append(np.where(gt_vol == output_class_value, 1, 0))

            gt_vol = np.stack(class_val_img_list, axis=-1)

        if self.normalize_ground_truth:
            gt_vol = gt_vol / norm

        # if not self.soft_labels:
        #     gt_vol = gt_vol.astype(np.uint8)

        self.ground_truth = gt_vol

    def _calc_metrics(self):
        self.performanceMetrics = PerformanceMetrics(y_true=self.ground_truth,
                                                     y_pred=self.predictions,
                                                     thr=self.classification_threshold,
                                                     enable_curves=self.enable_curves)

        self.measure_dict = self.performanceMetrics.measure_dict

class bboxPerformanceEvaluator:

    def __init__(self,
                 config: Union[PredictConfig, TrainConfig],
                 predicted_data: np.ndarray):
        self.config = config

        # this should be a [z, y, x, ch] volume
        self.predicted_data = predicted_data
        self.class_values = self.config.class_values
        self.n_output_values = self.config.n_output_classes

        if config.config_type == "predict":
            raise NotImplementedError

        self.gt_vol, self.bboxes = self._load_gt()
        self.measure_dict = self._calc_metrics()

    def _calc_metrics(self):
        metrics_dict = {}
        for idx, output_class_value in enumerate(self.class_values):
            class_gt_arr = np.array(())
            class_pred_arr = np.array(())
            for bbox_idx, bbox in enumerate(self.bboxes):
                predictions = self.predicted_data[bbox_idx, bbox[1]:bbox[3], bbox[0]:bbox[2], idx]
                predictions_flat = predictions.flatten()
                class_pred_arr = np.append(class_pred_arr, predictions_flat)

                gt = self.gt_vol[bbox_idx, bbox[1]:bbox[3], bbox[0]:bbox[2], idx]
                gt_flat = gt.flatten()
                class_gt_arr = np.append(class_gt_arr, gt_flat)

            pe = SingleVolumePerformanceEvaluator(self.config, predicted_data=class_pred_arr, gt_array=class_gt_arr)
            metrics_dict[output_class_value] = pe.measure_dict

        return metrics_dict


    def _load_gt(self):
        gt_img_paths = sorted(glob_imgs(self.config.ground_truth_path, mode="stack"))

        csv_paths = [fpath.parent.joinpath(fpath.name+".csv") for fpath in gt_img_paths]
        bboxes = [self._parse_csv(fpath) for fpath in csv_paths]

        img_list = []
        for fpath in gt_img_paths:
            img = skio.imread(str(fpath), plugin="pil")

            class_value_img_list = []
            for class_value in self.class_values:
                val_img = np.where(img == class_value, 1, 0)
                class_value_img_list.append(val_img)

            img = np.stack(class_value_img_list, axis=-1)
            img_list.append(img)

        gt_vol = np.stack(img_list, axis=0)

        return gt_vol, bboxes


    @staticmethod
    def _parse_csv(csv_path: Path) -> list:
        """return the first row of a csv file"""
        out_list = []
        with csv_path.open(mode="r") as infile:
            reader = csv.reader(infile)
            for row in reader:
                row_ints = [int(elem) for elem in row]
                out_list.append(row_ints)
        return out_list[0]


class H5PerformanceEvaluator:
    def __init__(self,
                 config: Union[PredictConfig, TrainConfig],
                 predict_pathlist: list):
        self.config = config

        self.predict_pathlist = predict_pathlist
        if config.config_type == "predict":
            raise NotImplementedError("Performance evaluation for prediction on h5 datasets is not yet supported")
        elif config.config_type == "training":
            self.normalize_ground_truth = config.normalize_masks

        self.measure_dict = self._calc_aggregated_metrics()

    def _calc_metrics(self) -> dict:
        metrics_dict = {}
        h5file = h5py.File(str(self.config.path_dict["test"]), "r")

        for idx, path in enumerate(self.predict_pathlist):
            ground_truth = h5file["data"][idx, 1, ...]
            if self.normalize_ground_truth:
                norm = np.iinfo(ground_truth.dtype).max
                ground_truth = ground_truth / norm
            with self.predict_pathlist[idx].open(mode="rb") as infile:
                prediction = pickle.load(infile)
            pe = SingleVolumePerformanceEvaluator(self.config, prediction, ground_truth)
            metrics_dict[str(idx)] = pe.measure_dict

        self.metrics_dict = metrics_dict
        return metrics_dict

    def _calc_aggregated_metrics(self) -> dict:
        metrics_dict = self.metrics_dict if hasattr(self, "metrics_dict") else self._calc_metrics()
        aggregated_dict = self._get_metric_aggregated_dict(metrics_dict)
        mean_aggregated_dict = {metric: np.mean(metric_array) for metric, metric_array in aggregated_dict.items()}
        return mean_aggregated_dict

    @staticmethod
    def _get_metric_aggregated_dict(sample_metric_dict) -> dict:
        first_elem = list(sample_metric_dict.keys())[0]
        metrics = list(sample_metric_dict[first_elem].keys())

        aggregated_dict = {metric: [] for metric in metrics}

        for sample_name, metric_dict in sample_metric_dict.items():
            for metric in metrics:
                aggregated_dict[metric].append(metric_dict[metric])

        return aggregated_dict


class MultiVolumePerformanceEvaluator:
    def __init__(self,
                 config: Union[PredictConfig, TrainConfig],
                 prediction_dict: dict):
        self.config = config
        self.prediction_dict = {key: np.squeeze(prediction) for key, prediction in prediction_dict.items()}
        self.ground_truth_mode = config.ground_truth_mode
        self.ground_truth_path = config.ground_truth_path

        if config.config_type == "predict":
            self.normalize_ground_truth = config.ground_truth_normalize
        elif config.config_type == "training":
            self.normalize_ground_truth = config.normalize_masks

        self.gt_dict = self._load_gt()

        self.metrics = self._calc_metrics()
        self.aggregated_metrics = self._calc_aggregated_metrics()

    def _load_gt(self) -> dict:
        self.ground_truth_volume_fpaths = glob_imgs(self.ground_truth_path, mode="stack", to_string=False)
        gt_fpath_names = [fpath.name for fpath in self.ground_truth_volume_fpaths]
        if set(self.prediction_dict.keys()) != set(gt_fpath_names):
            raise ValueError("GT masks not matching GT frames")

        gt_dict = {}
        for idx, gt_fpath in enumerate(self.ground_truth_volume_fpaths):
            label_centers = None
            volume_name = gt_fpath.name
            gt_volume, norm = load_volume(gt_fpath,
                                          ignore_last_channel=False,
                                          data_mode="stack",
                                          return_norm=True)
            if self.normalize_ground_truth and not self.config.binarize_gt:
                gt_volume = gt_volume / norm
            if self.config.binarize_gt:
                gt_volume = np.where(gt_volume > 0, 1, 0).astype(gt_volume.dtype)
            gt_dict[volume_name] = gt_volume

        return gt_dict

    def _calc_metrics(self) -> dict:
        metrics_dict = {}
        for name, prediction in self.prediction_dict.items():
            ground_truth = self.gt_dict[name]
            pe = SingleVolumePerformanceEvaluator(self.config, prediction, ground_truth)
            metrics_dict[name] = pe.measure_dict

        self.metrics_dict = metrics_dict
        return metrics_dict

    def _calc_aggregated_metrics(self) -> dict:
        metrics_dict = self.metrics_dict if hasattr(self, "metrics_dict") else self._calc_metrics()
        aggregated_dict = self._get_metric_aggregated_dict(metrics_dict)
        mean_aggregated_dict = {metric: np.mean(metric_array) for metric, metric_array in aggregated_dict.items()}
        return mean_aggregated_dict

    @staticmethod
    def _get_metric_aggregated_dict(sample_metric_dict) -> dict:
        first_elem = list(sample_metric_dict.keys())[0]
        metrics = list(sample_metric_dict[first_elem].keys())

        aggregated_dict = {metric: [] for metric in metrics}

        for sample_name, metric_dict in sample_metric_dict.items():
            for metric in metrics:
                aggregated_dict[metric].append(metric_dict[metric])

        return aggregated_dict


def PerformanceEvaluator(config: Union[TrainConfig, PredictConfig],
                         predicted_data: Union[np.ndarray, dict, list] = None) -> Union[
    MultiVolumePerformanceEvaluator, SingleVolumePerformanceEvaluator, H5PerformanceEvaluator, bboxPerformanceEvaluator]:
    data_mode = config.dataset_mode if config.config_type == "training" else config.data_mode
    if data_mode in ["single_images", "stack"]:
        # predicted_data should be an ndarray
        if config.use_bboxes:
            return bboxPerformanceEvaluator(config, predicted_data=predicted_data)
        else:
            return SingleVolumePerformanceEvaluator(config, predicted_data=predicted_data)
    elif data_mode == "multi_stack":
        # predicted_data should be a dict
        return MultiVolumePerformanceEvaluator(config, prediction_dict=predicted_data)
    elif data_mode == "h5_dataset":
        return H5PerformanceEvaluator(config, predict_pathlist=predicted_data)


class NodeMetrics:
    def __init__(
        self,
        predicted_centers: Tuple[tuple],
        ground_truth_centers: Tuple[tuple],
        resolution: Union[np.ndarray, tuple, list],
        detection_distance: int = 0.2
    ):
        self.predicted_centers = predicted_centers
        self.ground_truth_centers = ground_truth_centers
        self.resolution = np.array(resolution)
        self.detection_distance = detection_distance

        predicted_nodes = np.array(self.predicted_centers)
        predicted_nodes = predicted_nodes * resolution
        self.predicted_nodes = [("P-"+str(uuid.uuid4()),tuple(node)) for node in predicted_nodes]

        gt_nodes = np.array(self.ground_truth_centers)
        gt_nodes = gt_nodes * resolution
        self.gt_nodes = [("T-"+str(uuid.uuid4()),tuple(node)) for node in gt_nodes]

        self.graph = self._gen_graph(self.predicted_nodes, self.gt_nodes)
        self.graph = self._populate_edges(
            graph=self.graph,
            distance_fn=self._distance,
            max_dist=self.detection_distance,
        )

        self.true_positives, self.true_negatives, self.false_positives, self.false_negatives = self.get_confusion_matrix(
            self.graph,
            max_dist=self.detection_distance,
            distance_fn=self._distance
        )

        self.metrics = self.get_metrics_dict()


    def get_metrics_dict(self):
        tp = len(self.true_positives)
        tn = len(self.true_negatives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        
        precision =  tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 =  2*tp / (2*tp + fp +fn)
        jaccard = tp / (tp + fn + fp)
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "jaccard": jaccard
        }
        return metrics
    
    @staticmethod
    def _gen_graph(predicted_nodes: list, gt_nodes: list) -> nx.Graph:
        graph = nx.Graph()

        graph.add_nodes_from(predicted_nodes, type="predicted")
        graph.add_nodes_from(gt_nodes, type="ground_truth")

        return graph
    
    @staticmethod
    def _distance(arr_a: np.ndarray, arr_b: np.ndarray):
        return np.sqrt(sum((arr_a - arr_b)**2))

    @staticmethod
    def _populate_edges(
        graph: nx.Graph,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
        max_dist: float,
        epsilon: float = 1e-4):
        predicted_nodes_nx = [node for node, data in graph.nodes(data=True) if data["type"] == "predicted"]
        gt_nodes_nx = [node for node, data in graph.nodes(data=True) if data["type"]=="ground_truth"]
        
        graph.remove_edges_from(list(graph.edges()))
        for node_gt in gt_nodes_nx:
            for node_pred in predicted_nodes_nx:
                dist = distance_fn(np.array(node_gt[1]), np.array(node_pred[1]))
                if dist < max_dist:
                    w = 1.0/max(epsilon, dist)
                    graph.add_edge(node_gt, node_pred, weight=w)
                    
        return graph
    
    @staticmethod
    def get_confusion_matrix(graph: nx.Graph, max_dist: float, distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[list, list, list, list]:
        matching = nx.algorithms.max_weight_matching(graph, maxcardinality=False)
        
        matched_predictions = [node for match in matching for node in match if node[0][0] == "P"]
        matched_groundtruth = [node for match in matching for node in match if node[0][0] == "T"]

        unmatched_nodes = list(set(graph.nodes) - set(matched_groundtruth) - set(matched_predictions))
        
        false_positives = [node for node in unmatched_nodes if node[0][0] == "P"]
        false_negatives = [node for node in unmatched_nodes if node[0][0] == "T"]

        true_positives = [node for match in matching for node in match if node[0][0] == "P" and distance_fn(np.array(match[0][1]), np.array(match[1][1])) < max_dist / 2]
        true_negatives = list(set(graph.nodes) - set(false_positives) - set(false_negatives) - set(true_positives))

        return (true_positives, true_negatives, false_positives, false_negatives)
