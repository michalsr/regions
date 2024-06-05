import pickle
import os
from json import JSONEncoder
from typing import Dict, Optional
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
import torch
from typing import List, Dict, Any
import logging
import coloredlogs
from PIL import Image, ImageDraw
import warnings
import stat

def save_file(filename,data):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """

    parent_dir = os.path.dirname(filename)
    if parent_dir != '':
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    file_ext = os.path.splitext(filename)[1]
    
    if file_ext == ".npy":
        with open(filename, "wb+") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":

        with open(filename,'w+') as fopen:
            json.dump(data,fopen,indent=2)

    elif file_ext == ".yaml":
        with open(filename, "w+") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        # assume file is pickle
         with open(filename, "wb+") as fopen:
            pickle.dump(data, fopen)




def open_file(filename):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.txt':
        with open(filename,'r+') as fopen:
            data = fopen.readlines()
    elif file_ext in [".npy",".npz"]:
        data = np.load(filename,allow_pickle=True)
    elif file_ext == '.json':
        with open(filename,'r+') as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with open(filename,'r+') as fopen:
            data = yaml.load(fopen,Loader=yaml.FullLoader)
    else:
        # assume pickle
        with open(filename,"rb+") as fopen:
            data = pickle.load(fopen)
    return data
def intersect_and_union(
    pred_label,
    label,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
    reduce_pred_labels: bool = False,
):
    """Calculate intersection and Union.
    Args:
        pred_label (`ndarray`):
            Prediction segmentation map of shape (height, width).
        label (`ndarray`):
            Ground truth segmentation map of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        reduce_pred_labels (`bool`, *optional*, defaults to `False`):
            Do the same as `reduce_labels` but for prediction labels.
     Returns:
         area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    # turn into Numpy arrays
    pred_label = np.array(pred_label)
    label = np.array(label)

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    if reduce_pred_labels:
        pred_label[pred_label == 0] = 255
        pred_label = pred_label - 1
        pred_label[pred_label == 254] = 255

    mask = label != ignore_index
    #mask = np.not_equal(label, ignore_index)
    pred_label = pred_label[label!=ignore_index]

    label = label[label!= ignore_index]

    #label = np.array(label)[mask]

    intersect = pred_label[pred_label == label]

    area_intersect = np.histogram(intersect, bins=num_labels, range=(0, num_labels - 1))[0]
    area_pred_label = np.histogram(pred_label, bins=num_labels, range=(0, num_labels - 1))[0]
    area_label = np.histogram(label, bins=num_labels, range=(0, num_labels - 1))[0]

    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
    reduce_pred_labels: bool = False
):
    """Calculate Total Intersection and Union, by calculating `intersect_and_union` for each (predicted, ground truth) pair.
    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        reduce_pred_labels (`bool`, *optional*, defaults to `False`):
            Same as `reduce_labels` but for prediction labels.
     Returns:
         total_area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         total_area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         total_area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         total_area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
    total_area_union = np.zeros((num_labels,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
    total_area_label = np.zeros((num_labels,), dtype=np.float64)
    for result, gt_seg_map in tqdm(zip(results, gt_seg_maps), total=len(results)):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_labels, ignore_index, label_map, reduce_labels, reduce_pred_labels
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def mean_iou(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
    reduce_pred_labels: bool = False,
):
    """Calculate Mean Intersection and Union (mIoU).
    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        nan_to_num (`int`, *optional*):
            If specified, NaN values will be replaced by the number defined by the user.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        reduce_pred_labels (`bool`, *optional*, defaults to `False`):
            Same as `reduce_labels` but for prediction labels.
    Returns:
        `Dict[str, float | ndarray]` comprising various elements:
        - *mean_iou* (`float`):
            Mean Intersection-over-Union (IoU averaged over all categories).
        - *mean_accuracy* (`float`):
            Mean accuracy (averaged over all categories).
        - *overall_accuracy* (`float`):
            Overall accuracy on all images.
        - *per_category_accuracy* (`ndarray` of shape `(num_labels,)`):
            Per category accuracy.
        - *per_category_iou* (`ndarray` of shape `(num_labels,)`):
            Per category IoU.
    """
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
        results, gt_seg_maps, num_labels, ignore_index, label_map, reduce_labels, reduce_pred_labels
    )

    # compute metrics
    metrics = dict()

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label

    metrics["mean_iou"] = np.nanmean(iou)
    metrics["mean_accuracy"] = np.nanmean(acc)
    metrics["overall_accuracy"] = all_acc
    metrics["per_category_iou"] = iou
    metrics["per_category_accuracy"] = acc

    if nan_to_num is not None:
        metrics = dict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in metrics.items()}
        )

    return metrics

def split_gt_masks(gt_img:np.array, ignore_zero_label:bool):
    """
    Copy of function from sam_analysis/max_possible_iou.py. Splits instance mask into binary masks
    Used to process ADE20K instance annotations
    """
    labels = np.asarray([l for l in np.unique(gt_img) if l!= 0 or not ignore_zero_label])
    if any(labels < 0):
        raise ValueError('Label value less than zero detected. This is not supported.')

    if len(labels) == 0:
        err_str = 'Failed to split GT image into masks; no labels detected.'
        logger.error(err_str)
    if not ignore_zero_label:
        labels += 1
        gt_img += 1
    gt_masks = np.stack([(gt_img == l) * l for l in labels]) # (nlabels, h, w)

    return gt_masks, labels

