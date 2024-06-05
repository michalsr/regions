import numpy as np
import json
from PIL import Image, ImageColor
from pycocotools import mask as mask_utils
from tqdm import tqdm
import os
import scipy
import cv2
import pickle
import segmentation_utils as utils
import torch
import torchvision.transforms as T
import argparse
import torch.nn.functional as F
import logging 
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger)

def load_all_sam_regions(args):
    logger.info(f"Loading sam regions from {args.sam_dir}")
    image_id_to_sam = {}
    for f in tqdm(os.listdir(args.sam_dir)):
        if '.json' in f:
            sam_regions = utils.open_file(os.path.join(args.sam_dir,f))
            image_id_to_sam[f.replace('.json','')] = sam_regions
    return image_id_to_sam



def label_region(args,sam_region,annotation_map):
    sam_region_nonzero = np.where(sam_region != 0)
    # get pixel values from map

    pixel_values_in_region = annotation_map[sam_region_nonzero[0],sam_region_nonzero[1]].flatten()
    unique_pixels, pixel_counts = np.unique(pixel_values_in_region,return_counts=True)
    all_pixels_in_region = dict(zip(unique_pixels,pixel_counts))
    if args.num_classes ==151:
        start_class = 1
    else:
        start_class = 0

    # get total num of pixels
    num_pixels = sum(all_pixels_in_region.values())
    #check if any pixel is greater than certain percent value
    more_than_percent= [(pixel_val,pixel_count) for pixel_val,pixel_count in all_pixels_in_region.items() if all((pixel_count>((args.label_percent/100)*num_pixels),pixel_val>=start_class,pixel_val<=args.num_classes))]
    # initialize all as None

    initial_label  = {key: None for key in list(range(start_class,args.num_classes+1))}
    final_label = {}


    if len(more_than_percent)>0:
        max_idx = np.argmax(np.asarray([t[1] for t in more_than_percent]))
        max_pixel_class = [t[0] for t in more_than_percent][max_idx]
        # positive for that label

        final_label[max_pixel_class] = 1
        # negative for the rest
        for key in list(range(start_class,int(args.num_classes))):
            if key != max_pixel_class:
                final_label[key] = -1
    else:
        # all zero
        final_label = {key:0 for key in list(range(start_class,int(args.num_classes)))}
    return final_label


def label_all_regions(args):
    if len(os.listdir(args.sam_dir)) == 0:
        raise Exception(f"No sam regions found at {args.sam_dir}")
    image_id_to_sam = load_all_sam_regions(args)
    if args.gt_regions:
        # each region already has a label
        for image_id,entry in tqdm(image_id_to_sam.items()):
            region_to_label = []
            for region in entry:
                gt_labels = {}
                gt_mask = region['segmentation']
                labels = {str(region['label']):1}
                gt_labels['labels'] = labels
                region_to_label.append(gt_labels)
            utils.save_file(os.path.join(args.region_labels,image_id),region_to_label)
    else:
        all_annotations = os.listdir(args.annotation_dir)
        annotations_minus_sam = {ann for ann in all_annotations if ann.replace('.png', '') not in image_id_to_sam}

        if len(annotations_minus_sam) > 0:
            logger.info(f"Warning: {len(annotations_minus_sam)} annotations not found in SAM regions : {annotations_minus_sam}")
            logger.info("Restricting annotations to those with corresponding SAM regions")
            all_annotations = [ann for ann in all_annotations if ann not in annotations_minus_sam]
    

        for i,ann in enumerate(tqdm(all_annotations,desc='Label Features',total=len(all_annotations))):
            region_to_label = []
            annotation_map =np.array(Image.open(os.path.join(args.annotation_dir,ann)),dtype=np.int64)

            sam_regions = image_id_to_sam[ann.replace('.png','')]
            for region in sam_regions:
                sam_labels = {}
                sam_labels['region_id'] = region['region_id']
                sam_mask = mask_utils.decode(region['segmentation'])
                labels = label_region(args,sam_mask,annotation_map)
                sam_labels['labels'] = labels
                region_to_label.append(sam_labels)
            utils.save_file(os.path.join(args.region_labels,ann.replace('.png','.pkl')),region_to_label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region_labels",
        type=str,
        default=None,
        help="Location to store ground truth of label regions",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Location of per-pixel annotations",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="Location of SAM regions"
    )
    parser.add_argument(
        "--label_percent",
        type=int,
        default=50,
        help="Percent of pixels within a region that need to belong to the same class before region is assigned that label"
    )
   
    args = parser.parse_args()
    if args.num_classes != 151 and args.num_classes!= 21:
        raise ValueError('ADE should have 151 and Pascal VOC should have 21')
    label_all_regions(args)