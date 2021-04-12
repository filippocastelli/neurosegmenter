import numpy as np
import sklearn.metrics as skmetrics
import logging

from neuroseg.utils import load_volume, glob_imgs


def PerformanceEvaluator(config,
                         predicted_data=None):
    data_mode = config.dataset_mode if config.config_type == "training" else config.data_mode
    if data_mode in ["single_images", "stack"]:
        return SingleVolumePerformanceEvaluator(config, predicted_data=predicted_data)
    elif data_mode == "multi_stack":
        return MultiVolumePerformanceEvaluator(config, prediction_dict=predicted_data)

class MultiVolumePerformanceEvaluator:
    
    def __init__(self, config, prediction_dict):
        self.config = config
        self.prediction_dict = {key: np.squeeze(prediction) for key, prediction in prediction_dict.items()}
        self.ground_truth_mode = config.ground_truth_mode
        self.ground_truth_path = config.ground_truth_path
        self.gt_dict = self._load_gt()
        
        self.measure_dict = self._calc_aggregated_metrics()
        
        
    def _load_gt(self):
        self.ground_truth_volume_fpaths = glob_imgs(self.ground_truth_path, mode="stack", to_string=False)
        gt_fpath_names = [fpath.name for fpath in self.ground_truth_volume_fpaths]
        if set(self.prediction_dict.keys()) != set(gt_fpath_names):
            raise ValueError("GT masks not matching GT frames")
            
        gt_dict = {}
        for idx, gt_fpath in enumerate(self.ground_truth_volume_fpaths):
            volume_name = gt_fpath.name
            gt_volume = load_volume(gt_fpath,
                    drop_last_dim=False,
                    expand_last_dim=False,
                    squeeze=True,
                    data_mode="stack")
            norm_constant = np.iinfo(gt_volume.dtype).max
            gt_volume = gt_volume / norm_constant
            gt_dict[volume_name] = gt_volume.astype(np.uint8)
            
        return gt_dict
        
    def _calc_metrics(self):
        metrics_dict = {}
        for name, prediction in self.prediction_dict.items():
            ground_truth = self.gt_dict[name]
            pe = SingleVolumePerformanceEvaluator(self.config,prediction, ground_truth)
            metrics_dict[name] = pe.measure_dict
            
        self.metrics_dict = metrics_dict
        return metrics_dict   

    def _calc_aggregated_metrics(self):
        metrics_dict = self.metrics_dict if hasattr(self, "metrics_dict") else self._calc_metrics()
        aggregated_dict = self._get_metric_aggregated_dict(metrics_dict)
        mean_aggregated_dict = {metric: np.mean(metric_array) for metric, metric_array in aggregated_dict.items()}
        return mean_aggregated_dict
                    
    @staticmethod
    def _get_metric_aggregated_dict(sample_metric_dict):
        first_elem = list(sample_metric_dict.keys())[0]
        metrics = list(sample_metric_dict[first_elem].keys())
        
        aggregated_dict = {metric: [] for metric in metrics}
        
        for sample_name, metric_dict in sample_metric_dict.items():
            for metric in metrics:
                aggregated_dict[metric].append(metric_dict[metric])
                
                
        return aggregated_dict
                   
        
class SingleVolumePerformanceEvaluator:
    
    def __init__(self, config, predicted_data=None, gt_array=None):
        self.config = config

        if (gt_array is None):
            self.ground_truth_mode = config.ground_truth_mode
            self.ground_truth_path = self._get_gt_path()
            self._load_gt()
        else:
            self.ground_truth = gt_array
            
        self._preprocess_pred(predicted_data)
            
        self.classification_threshold = config.pe_classification_threshold
        self.enable_curves = config.pe_enable_curves
        self._calc_metrics()
        
        
    def _get_gt_path(self):
        if self.ground_truth_mode == "stack":
            return glob_imgs(self.config.ground_truth_path, mode="stack", to_string=False)[0]
        elif self.ground_truth_mode == "single_images":
            return self.config.ground_truth_path
        else:
            raise NotImplementedError(self.ground_truth_mode)
            
    def _load_data(self, pred_array):
        self._load_gt()
        self._load_predictions(pred_array)
        
    def _preprocess_pred(self, pred_array):
        if len(pred_array.shape) > len(self.ground_truth.shape):
            pred_array = np.squeeze(pred_array)
        
        assert(pred_array.shape == self.ground_truth.shape), "GT and predictions have different number of channels" 
        
        self.predictions = pred_array
        
    def _load_gt(self):
        gt_vol = (load_volume(self.ground_truth_path,
                             drop_last_dim=False,
                             expand_last_dim=False,
                             squeeze=True,
                             # data_mode=self.ground_truth_mode) / 255
                             data_mode=self.ground_truth_mode) / 255).astype(np.uint8)
        self.ground_truth = gt_vol
        
    def _calc_metrics(self):
        
        self.performanceMetrics = PerformanceMetrics(y_true = self.ground_truth,
                                                     y_pred = self.predictions,
                                                     thr=self.classification_threshold,
                                                     enable_curves=self.enable_curves)
        
        self.measure_dict = self.performanceMetrics.measure_dict
    
                 
class PerformanceMetrics:
    """general class for performance metrics evaluation"""
    def __init__(self, y_true, y_pred, thr, enable_curves=False):
        
        self.enable_curves = enable_curves
        self.y_true = np.copy(y_true)
        self.y_pred = np.copy(y_pred)
        self.thr = thr
        
        self.y_pred_fuzzy = y_pred
        self.y_true_fuzzy = y_true
        
        self.fuzzy_intersection = np.sum(self.y_pred_fuzzy.flatten() * self.y_true_fuzzy.flatten())
        self.fuzzy_summation = np.sum(self.y_pred_fuzzy.flatten()) + np.sum(self.y_true_fuzzy.flatten())
        self.fuzzy_union = self.fuzzy_summation - self.fuzzy_intersection
        
        self.y_pred = self.threshold_array(self.y_pred, thr, to_bool=True)
        self.y_true = self.threshold_array(self.y_true, thr, to_bool=True)

        self.tp, self.fp, self.tn, self.fn = self.cardinal_metrics()

        self.specificity = self.crisp_specificity()
        self.recall = self.crisp_recall()
        self.precision = self.crisp_precision()
        self.false_negative_ratio = self.crisp_false_negative_ratio()
        self.fallout = self.crisp_fallout()

        self.measure_dict = self.create_dict()

    def cardinal_metrics(self, sum_elems=True):
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

        if sum_elems == True:
            return (
                np.sum(true_positive),
                np.sum(false_positive),
                np.sum(true_negative),
                np.sum(false_negative),
            )
        else:
            return true_positive, false_positive, true_negative, false_negative
        
    def fuzzy_dice(self):
        """Dice coefficient for fuzzy segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (6)
        """
        return 2 * self.fuzzy_intersection / self.fuzzy_summation
    
    def fuzzy_jaccard(self):
        """Jaccard index for fuzzy segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (7)
        """
        return self.fuzzy_intersection / self.fuzzy_union
    
    @staticmethod
    def threshold_array(arr, thr=0.5, to_bool=False):
        arr[arr >= thr] = 1
        arr[arr < thr] = 0

        if to_bool == True:
            return np.array(arr, dtype=bool)
        else:
            return arr

    def crisp_jaccard(self):
        """Jaccard index for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (7)
        """
        return self.tp / (self.tp + self.fp + self.fn)

    def crisp_dice(self):
        """Dice coefficient for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (6)
        """
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    # recall/sensitivity
    def crisp_recall(self):
        """Recall or Sensitivity or TPR for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (10)
        """
        return (self.tp) / (self.tp + self.fn)

    # specificity/true negative ratio
    def crisp_specificity(self):
        """Specificity or TNR for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (11)
        """
        return (self.tn) / (self.tn + self.fp)

    # fallout/false positive ratio
    def crisp_fallout(self):
        """Fallout or FPR for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (12)
        """
        return 1 - self.specificity

    def crisp_false_negative_ratio(self):
        """FNR for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (13)
        """
        return 1 - self.recall

    # precision/positive predictive ratio
    def crisp_precision(self):
        """Precision or PPV for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (14)
        """
        return (self.tp) / (self.tp + self.fp)

    def crisp_auc(self):
        """Estimator of Area under ROC 
        ROC is defined as TPR vs FPR plot, AUC here is calculated as area of the 
        trapezoid defined by the measurement opint of the lines TPR=0 nad FPR=1.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (47)
        """
        return 1 - ((self.fallout + self.false_negative_ratio) / 2)

    def crisp_f1_score(self):
        """F1-score or FMS1 for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (16)
        """
        return (2 * self.precision * self.recall) / (self.precision + self.recall)
    
    def sk_f1_score(self):
        """F1-score implemented by scikit-learn"""
        return skmetrics.f1_score(self.y_true.flatten(), self.y_pred.flatten())
    
    def sk_roc_curve(self):
        """ROC curve
        Implemented by scikit-learn"""
        logging.info("Calculating ROC curve")
        fpr, tpr, thresholds = skmetrics.roc_curve(y_true=self.y_true_fuzzy.flatten(),
                                                   y_score=self.y_pred_fuzzy.flatten())
        
        roc_dict = {"fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds}
        return roc_dict
    
    def sk_roc_auc_score(self):
        """Area under ROC curve
        Implemented by scikit-learn"""
        logging.info("Calculating PR curve")
        flatten_y_true = self.y_true_fuzzy.flatten()
        flatten_y_pred = self.y_pred_fuzzy.flatten()
        
        return skmetrics.roc_auc_score(y_true=flatten_y_true,
                                       y_score=flatten_y_pred)
    
    def sk_pr_curve(self):
        """Precision Recall curve
        Implemented by scikit-learn"""
        precision, recall, threhsolds = skmetrics.precision_recall_curve(y_true=self.y_true_fuzzy.flatten(),
                                                                         probas_pred=self.y_pred_fuzzy.flatten())
        
        pr_dict = {"precision": precision,
                   "recall": recall,
                   "thresholds": threhsolds}
        return pr_dict

    def create_dict(self):
        crisp_jaccard =  self.crisp_jaccard()
        crisp_dice = self.crisp_dice()
        crisp_recall = self.crisp_recall()
        crisp_specificity = self.crisp_specificity()
        crisp_fallout = self.crisp_fallout()
        crisp_false_negative_ratio = self.crisp_false_negative_ratio()
        crisp_precision = self.crisp_precision()
        crisp_auc = self.crisp_auc()
        crisp_f1_score = self.crisp_f1_score()
        sk_f1_score  =  self.sk_f1_score()
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
            "fuzzy_dice" : fuzzy_dice,
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

    def get_metrics(self):
        return self.measure_dict