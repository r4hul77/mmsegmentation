import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable

from mmseg.registry import METRICS

@METRICS.register_module()
class LARSMetric(BaseMetric):
    """LARS evaluation metrics implementation.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. Defaults to False.
        prefix (str, optional): The prefix added in the metric names to
            disambiguate homonymous metrics of different evaluators. Defaults
            to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples."""
        for data_sample in data_samples:
            pred_seg = data_sample['pred_sem_seg']['data'].squeeze()
            gt_seg = data_sample['gt_sem_seg']['data'].squeeze().to(pred_seg)

            self.results.append({
                'pred_seg': pred_seg,
                'gt_seg': gt_seg
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        metrics = {
            'water_edge_accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'iou': []
        }

        for result in results:
            pred_seg = result['pred_seg']
            gt_seg = result['gt_seg']

            # Water-edge segmentation accuracy
            water_edge_accuracy = self._compute_water_edge_accuracy(pred_seg, gt_seg)
            metrics['water_edge_accuracy'].append(water_edge_accuracy)

            # Dynamic obstacle detection metrics
            precision, recall, f1_score = self._compute_detection_metrics(pred_seg, gt_seg)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1_score)

            # Mean IoU
            iou = self._compute_mean_iou(pred_seg, gt_seg)
            metrics['iou'].append(iou)

        # Aggregate metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        avg_metrics['Q'] = avg_metrics['iou'] * avg_metrics['f1_score']

        # Print detailed metrics
        self._print_metrics(avg_metrics)

        return avg_metrics

    @staticmethod
    def _compute_water_edge_accuracy(pred_seg, gt_seg):
        # Define a narrow belt around the ground-truth water-static-obstacle boundary
        boundary_mask = LARSMetric._get_boundary_mask(gt_seg)
        correct_pixels = (pred_seg[boundary_mask] == gt_seg[boundary_mask]).sum()
        total_pixels = boundary_mask.sum()
        return correct_pixels / total_pixels if total_pixels > 0 else 0

    @staticmethod
    def _compute_detection_metrics(pred_seg, gt_seg):
        # Compute precision, recall, and F1 score
        gt_obstacles = (gt_seg == 0)
        pred_obstacles = (pred_seg == 0)

        true_positive = (pred_obstacles & gt_obstacles).sum()
        false_positive = (pred_obstacles & ~gt_obstacles).sum()
        false_negative = (~pred_obstacles & gt_obstacles).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score

    @staticmethod
    def _compute_mean_iou(pred_seg, gt_seg):
        # Compute mean IoU
        intersect = (pred_seg & gt_seg).sum()
        union = (pred_seg | gt_seg).sum()
        return intersect / union if union > 0 else 0

    @staticmethod
    def _get_boundary_mask(gt_seg):
        # Define boundary mask logic (e.g., dilation/erosion)
        return torch.zeros_like(gt_seg, dtype=bool)  # Placeholder

    @staticmethod
    def _print_metrics(metrics):
        table = PrettyTable()
        table.field_names = ['Metric', 'Value']
        for key, value in metrics.items():
            table.add_row([key, value])
        print_log('\n' + table.get_string(), logger=MMLogger.get_current_instance())
