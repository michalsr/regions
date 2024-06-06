'''
Converts SLIC assignment matrices (or stacked masks from any source) to pseudo-SAM dicts usable by process_regions.py.
'''
# %%
from tqdm import tqdm
import json
import pickle
import os
from pycocotools import mask as mask_utils
import numpy as np
import argparse
def stacked_masks_to_sam_dicts(stacked_masks: np.ndarray, image_id: str):
    '''
    Creates a list of SAM dicts for use by process_regions.py from a stack of binary masks.
    Args:
        stacked_masks (np.ndarray): Binary tensor of stacked SAM masks. (n,h,w).
        image_id (str): Image ID of the image the SAM masks came from.
    Returns:
        sam_dicts (list): List of dicts as if output by sam.py with the properties essential for process_regions.py.
    '''
    # Following process_regions.py, we need the following fields for each split mask:
    # region_id: {image_id}_region_{i} from sam.py
    # area: area of the region (sum of binary mask)
    # segmentation: {
    #  "size": [h, w],
    # "counts": RLE-encoded binary mask
    # }
    segmentations = mask_utils.encode(np.asfortranarray(stacked_masks.astype(np.uint8).transpose(1, 2, 0))) # List of dicts
    for segmentation in segmentations:
        segmentation['counts'] = segmentation['counts'].decode('utf-8') # Decode to string to save as JSON

    return [
        {
            'region_id': f'{image_id}_region_{i}',
            'area': mask.sum().item(),
            'segmentation': segmentation
        } for i, (mask, segmentation) in enumerate(zip(stacked_masks, segmentations))
    ]
def assignment_to_sam_dicts(assignment: np.ndarray, image_id: str):
    '''
    Creates a list of SAM dicts from a SLIC image assignment matrix.

    Args:
        assignment (np.ndarray): Int array of SLIC superpixel assignments. Every pixel has an int assignment from 0 to n_regions. (h,w)
        image_id (str): Image ID of the image the SLIC assignment came from.

    Returns:
        sam_dict (list): List of dicts as if output by sam.py with the properties essential for process_regions.py.
    '''
    masks = np.stack([
        assignment == val
        for val in np.unique(assignment)
    ])

    return stacked_masks_to_sam_dicts(np.stack(masks), image_id)
def process_slic_path(slic_path: str, out_dir: str):
    '''
    Creates a SAM dict for each SLIC region in a SLIC assignment matrix and saves them to a JSON file.
    Args:
        slic_path (str): Path to the pickle with the SLIC assignment matrix output by gen_superpixels.py
        out_dir (str): Directory to save the SAM dicts to.
    '''
    with open(slic_path, 'rb') as f:
        assignment = pickle.load(f)['assignment']

    image_id = os.path.splitext(os.path.basename(slic_path))[0]
    sam_dicts = assignment_to_sam_dicts(assignment, image_id)

    with open(os.path.join(out_dir, f'{image_id}.json'), 'w') as f:
        json.dump(sam_dicts, f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str)
    parser.add_argument("--output_dir",type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    slic_paths = [os.path.join(args.input_dir, basename) for basename in sorted(os.listdir(args.input_dir))]
    for slic_path in tqdm(slic_paths):
            process_slic_path(slic_path, args.output_dir)


