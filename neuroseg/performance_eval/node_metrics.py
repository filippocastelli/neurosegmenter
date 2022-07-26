from pathlib import Path, PosixPath
from typing import Tuple, Union, Callable, Dict
import uuid

import networkx as nx
import numpy as np
import pandas as pd
from skimage import io as skio
import zetastitcher

from neuroseg.config import PredictConfig, TrainConfig
from neuroseg.utils import IntegerShearingCorrect
from neuroseg.utils import glob_imgs


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


class InstanceSegmentationPerformanceEvaluator:
    def __init__(
        self,
        ground_truth: Union[Dict[str, Path], Dict[str, np.ndarray]] = None,
        instance_segmentation: Union[Dict[str, Path], Dict[str, np.ndarray]] = None,
        resolution: float = (1., 1., 1.),
        config: Union[PredictConfig, TrainConfig] = None,
        max_dist: float = None,
    ):
        self.config = config
        self.resolution = tuple(resolution)
        self.max_dist = max_dist

        if self.config is not None:
            self.resolution = tuple(self.config.instance_performance_evaluation_resolution)
            self.max_dist = float(self.config.instance_performance_evaluation_max_dist)
        
        if ground_truth is None:
            gt_paths = sorted(glob_imgs(self.config.ground_truth_path, mode="stack"))
            self.ground_truth = {fpath.stem: fpath for fpath in gt_paths}
        else:
            self.ground_truth = {}

        gt_keys = set(self.ground_truth.keys())
        
        if instance_segmentation is not None and instance_segmentation != {}:
            instance_segmentation_keys = set(instance_segmentation.keys())

            if gt_keys != instance_segmentation_keys:
                print("instance_segmentation_keys:")
                print(instance_segmentation_keys)
                print("ground_truth_keys:")
                print(gt_keys)
                raise ValueError("ground_truth and instance_segmentation have different keys")

            self.metrics_dict = self._get_node_metrics(
                instance_segmentation=instance_segmentation,
                ground_truth=self.ground_truth)

            self.aggregated_metrics_dict = self._get_aggregated_metrics(self.metrics_dict)
        else:
            self.metrics_dict = {}
            self.aggregated_metrics_dict = {}

    @staticmethod
    def _load_stack(stack_path: Path) -> np.ndarray:
        stack = zetastitcher.InputFile(stack_path).whole()
        #stack = skio.imread(stack_path, plugin="pil")
        assert type(stack) is np.ndarray, f"something went wrong during loading of stack {stack_path}"
        return stack

    def _get_node_metrics(self,
        instance_segmentation: dict,
        ground_truth: dict) -> dict:
        node_metrics = {}

        for stack_name, prediction_stack in instance_segmentation.items():
            if type(prediction_stack) is PosixPath:
                prediction_stack = self._load_stack(prediction_stack)

            ground_truth_stack = ground_truth[stack_name]
            if type(ground_truth_stack) is PosixPath:
                ground_truth_stack = self._load_stack(ground_truth_stack)
            
            ground_truth_centers = self._get_centers(ground_truth_stack)
            predicted_centers = self._get_centers(prediction_stack)

            # apply shearing to centers
            ground_truth_centers = [self._apply_shearing(center, shearing_factor_x=-7) for center in ground_truth_centers]
            predicted_centers = [self._apply_shearing(center, shearing_factor_x=-7) for center in predicted_centers]

            nm = NodeMetrics(
                predicted_centers=tuple(predicted_centers),
                ground_truth_centers=tuple(ground_truth_centers),
                resolution=self.resolution,
                detection_distance=self.max_dist
            )

            node_metrics[stack_name] = nm.metrics
        
        return node_metrics

    @staticmethod
    def _apply_shearing(center_coords: np.ndarray, shearing_factor_x: float) -> np.ndarray:
        return (center_coords[0], center_coords[1], center_coords[2] + center_coords[0]*shearing_factor_x) 
        
    @staticmethod
    def _get_aggregated_metrics(node_metrics: Dict[str, dict]) -> dict:
        metric_dicts = list(node_metrics.values())
        keys = list(metric_dicts[0].keys())

        aggregated_metric_dict = {}

        for key in keys:
            metric_list = [metric_dict[key] for metric_dict in metric_dicts]
            metric_mean = np.mean(np.array(metric_list))
            aggregated_metric_dict[key] = metric_mean

        return aggregated_metric_dict

    
    @staticmethod
    def _get_centers(
            stack: np.ndarray,
            resolution: Union[np.ndarray, tuple, list] = (1., 1., 1.)) -> np.ndarray:
        labels = list(np.unique(stack))
        if 0 in labels:
            labels.remove(0) # removing background label
        centers = [np.mean(np.argwhere(stack == label), axis=0) for label in labels]
        centers = np.array(centers) * np.array(resolution)
        return centers




