'''
Script to find SLIC superpixels which have nonempty intersection with unmasked
regions of an image (i.e. regions which are not masked out by the SAM segmentation).
'''
import argparse
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import json
import pickle
from PIL import Image
from pycocotools import mask as mask_utils
from gen_superpixels import img_from_superpixels
from sam_analysis.visualize_sam import image_from_masks, show, masks_to_boundaries
from tqdm import tqdm
from einops import rearrange
from to_sam_format import stacked_masks_to_sam_dicts
import torchvision
import coloredlogs, logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

def get_unmasked_slic_regions(slic_assignment: torch.IntTensor, sam_masks: torch.BoolTensor, min_proportion: float = 0., min_pixels: int = 0) -> torch.BoolTensor:
    '''
    Get the SLIC regions which have nonempty intersection with unmasked regions of an image.

    Args:
        slic_assignment (torch.IntTensor): Tensor of SLIC superpixel assignments. (h,w).
        sam_masks (torch.IntTensor): Binary tensor of stacked SAM masks. (n,h,w).
        min_proportion (float): Minimum proportion of a SLIC region which must be in the intersection

    Returns:
        slic_regions (torch.Tensor): Binary tensor of SLIC regions with nonzero intersection. (n,h,w).
    '''
    # Collapse SAM masks into a single binary mask indicating where regions are not
    any_sam_mask = sam_masks.any(dim=0) # (h,w)
    no_sam_mask = ~any_sam_mask # (h,w)

    slic_vals_in_no_sam_mask = set(slic_assignment[no_sam_mask].tolist())

    # Keep only the SLIC regions which have a minimum proportion of their pixels in the intersection
    slic_regions = []
    for val in slic_vals_in_no_sam_mask:
        slic_region = slic_assignment == val
        intersection = slic_region & no_sam_mask

        n_pixels_in_intersection = intersection.sum()
        proportion_in_intersection = n_pixels_in_intersection / slic_region.sum()

        if n_pixels_in_intersection >= min_pixels and proportion_in_intersection >= min_proportion:
            slic_regions.append(slic_region)

    if len(slic_regions) == 0:
        return torch.zeros_like(slic_assignment).unsqueeze(0).bool() # (1,h,w)

    return torch.stack(slic_regions) # (n,h,w)

def get_merged_sam_and_slic_regions(unmasked_slic_regions: torch.BoolTensor, sam_masks: torch.BoolTensor):
    '''
    Merge the SLIC regions which have nonempty intersection with unmasked regions of an image, then stack them with the
    SAM masks.
    '''
    # Check if all SLIC regions got filtered out
    if len(unmasked_slic_regions) == 1 and not unmasked_slic_regions.any():
        return sam_masks

    intersected_slic_regions = (unmasked_slic_regions * sam_masks.any(dim=0).logical_not()).bool() # Intersect with unmasked regions; (n_slic,h,w)

    return torch.cat([sam_masks, intersected_slic_regions]) # (n_slic + n_sam,h,w)

def image_from_slic_and_sam(img: torch.Tensor, sam_masks, slic_regions: torch.Tensor, show_boundaries=False):
    '''
    Args:
        img (torch.Tensor): (c, h, w)
        slic_regions (torch.Tensor): (n, h, w)
    '''
    # Visualize the SAM masks
    img = image_from_masks(sam_masks, superimpose_on_image=img)

    # Number the regions and collapse
    if show_boundaries:
        slic_regions = masks_to_boundaries(slic_regions)

    return image_from_masks(slic_regions, combine_as_binary_mask=True, superimpose_on_image=img)
def merge_all(args):
    os.makedirs(args.sam_output_dir, exist_ok=True)
    for mask_basename in tqdm(sorted(os.listdir(args.sam_dir))):
        mask_path = os.path.join(args.sam_dir, mask_basename) # JSON
        slic_path = os.path.join(args.slic_dir, os.path.splitext(mask_basename)[0] + '.pkl')
        if not os.path.exists(slic_path):
            logger.warning(f'No SLIC assignment found at {slic_path}. Skipping.')
            continue
        with open(mask_path, 'r') as f:
            sam_masks = json.load(f)

        sam_masks = torch.tensor(
            np.stack([mask_utils.decode(mask['segmentation']) for mask in sam_masks])
        , dtype=torch.bool, device=args.device) # (n,h,w)

        # Load SLIC regions
        with open(slic_path, 'rb') as f:
            slic_assignment = torch.tensor(pickle.load(f)['assignment'], device=args.device)

        # Merge SLIC with SAM
        unmasked_slic_regions = get_unmasked_slic_regions(slic_assignment, sam_masks, min_proportion=args.min_proportion, min_pixels=args.min_pixels)
        merged_slic_and_sam_regions = get_merged_sam_and_slic_regions(unmasked_slic_regions, sam_masks).cpu()

        # Dump merged regions to SAM output file
        image_id = os.path.splitext(mask_basename)[0]
        sam_dicts = stacked_masks_to_sam_dicts(merged_slic_and_sam_regions.numpy(), image_id)

        with open(os.path.join(args.sam_output_dir, f'{image_id}.json'), 'w') as f:
            json.dump(sam_dicts, f)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_dir",type=str,help="Path to folder of SAM masks")
    parser.add_argument("--slic_dir",type=str,help="Path to formatted SLIC folder")
    parser.add_argument("--img_dir",type=str,help="Image dir")
    parser.add_argument("--sam_output_dir",type=str,help="Path to save SAM+SLIC")
    parser.add_argument("--min_proportion",type=int,default=0)
    parser.add_argument("--min_pixels",type=int,default=300)
    parser.add_argument("--device",type=str,default='cpu')
    args = parser.parse_args()
    merge_all(args)